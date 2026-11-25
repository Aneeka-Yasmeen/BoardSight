import argparse
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--params", required=True)
a = ap.parse_args()
p = {k: np.array(v, np.float32) for k, v in json.load(open(a.params)).items()}
m = onnx.load(a.model)
g = m.graph
if any(o.name == "scene" for o in g.output):
    raise SystemExit("board check is already in this model")


def const(name, value):
    g.initializer.append(numpy_helper.from_array(np.asarray(value, np.float32), name))
    return name


def node(op, inputs, name, **attr):
    g.node.append(helper.make_node(op, inputs, [name], name="scene/" + name, **attr))
    return name


mid, deep = "/model.6/cv2/act/Mul_output_0", "/model.10/cv2/act/Mul_output_0"
mean6 = node("ReduceMean", [mid], "sc_mean6", axes=[2, 3], keepdims=0)
mean10 = node("ReduceMean", [deep], "sc_mean10", axes=[2, 3], keepdims=0)
square = node("Mul", [mid, mid], "sc_square")
mean_square = node("ReduceMean", [square], "sc_mean_square", axes=[2, 3], keepdims=0)
square_mean = node("Mul", [mean6, mean6], "sc_square_mean")
variance = node("Sub", [mean_square, square_mean], "sc_variance")
positive = node("Max", [variance, const("sc_zero", 0.0)], "sc_positive")
std6 = node("Sqrt", [positive], "sc_std6")
feats = node("Concat", [mean6, mean10, std6], "sc_features", axis=1)
centred = node("Sub", [feats, const("sc_mu", p["mu"][None])], "sc_centred")
scaled = node("Div", [centred, const("sc_sd", p["sd"][None])], "sc_scaled")
clipped = node("Clip", [scaled, const("sc_lo", -6.0), const("sc_hi", 6.0)], "sc_clipped")
node("Gemm", [clipped, const("sc_w", p["w"][:, None]), const("sc_b", p["b"].reshape(1, 1))], "scene")
g.output.append(helper.make_tensor_value_info("scene", TensorProto.FLOAT, [1, 1]))
del m.metadata_props[:]
m.doc_string = ""
g.doc_string = ""
onnx.checker.check_model(m)
onnx.save(m, a.model)
print("board check added, metadata removed")

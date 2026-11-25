package org.pcbqc.edge

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import org.pcbqc.edge.ui.EdgeModel
import org.pcbqc.edge.ui.EdgeScreen

class MainActivity : ComponentActivity() {
    private val model: EdgeModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(SystemBarStyle.dark(0), SystemBarStyle.dark(0))
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent { EdgeScreen(model) }
    }

    override fun onStart() {
        super.onStart()
        model.enter()
    }

    override fun onStop() {
        if (!isChangingConfigurations) model.leave()
        super.onStop()
    }
}

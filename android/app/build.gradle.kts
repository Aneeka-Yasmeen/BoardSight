plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val abis = (project.findProperty("abis") as String? ?: "arm64-v8a").split(",")

android {
    namespace = "org.pcbqc.edge"
    compileSdk = 36
    buildToolsVersion = "36.0.0"

    defaultConfig {
        applicationId = "org.pcbqc.edge"
        minSdk = 28
        targetSdk = 36
        versionCode = 7
        versionName = "1.6"
        ndk { abiFilters += abis }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures { compose = true }
    androidResources { noCompress += listOf("onnx") }
    packaging {
        jniLibs {
            useLegacyPackaging = false
            pickFirsts += "**/libc++_shared.so"
        }
    }
    testOptions { unitTests.isReturnDefaultValues = true }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
    val bom = platform("androidx.compose:compose-bom:2025.09.00")
    implementation(bom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.11.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.3")
    implementation("androidx.camera:camera-camera2:1.5.0")
    implementation("androidx.camera:camera-lifecycle:1.5.0")
    implementation("androidx.camera:camera-view:1.5.0")
    implementation("androidx.exifinterface:exifinterface:1.4.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
    implementation("org.opencv:opencv:4.12.0")
    implementation("org.pytorch:executorch-android:1.5.1")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}

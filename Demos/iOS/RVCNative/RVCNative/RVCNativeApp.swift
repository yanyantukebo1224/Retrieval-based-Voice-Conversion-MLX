import SwiftUI
import RVCNativeFeature
import Metal
import Accelerate

@main
struct RVCNativeApp: App {
    init() {
        // App 最優先起動タイミングでログ転送・記録を開始
        _ = ConsoleLogRedirector.shared
        print("=== RVCNative App Started ===")
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

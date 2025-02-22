import React, { useState, useCallback } from "react";
import { useONNXModel } from "../lib/useONNXModel";

const OnnxDoubleModel: React.FC = () => {
  const [inputValue, setInputValue] = useState<string>("1,2,3,4");

  const { device, loading, status, result, inferenceTime, runInference } =
    useONNXModel({
      workerPath: "../lib/worker_onnx_double.ts",
      onError: (error) => console.error("Model error:", error),
    });

  // Handle form submission
  const handleRunInference = useCallback(() => {
    try {
      const input = inputValue.split(",").map((val) => parseFloat(val.trim()));

      if (input.some(isNaN)) {
        throw new Error("Invalid input. Please enter comma-separated numbers.");
      }

      runInference(input);
    } catch (error) {
      console.error("Input error:", error);
    }
  }, [inputValue, runInference]);

  // Get status badge color based on current status
  const getStatusBadgeColor = useCallback(() => {
    if (status.includes("Error"))
      return "bg-red-100 text-red-800 border-red-200";
    if (status.includes("warnings"))
      return "bg-yellow-100 text-yellow-800 border-yellow-200";
    if (status.includes("complete") || status.includes("successfully"))
      return "bg-green-100 text-green-800 border-green-200";
    if (loading) return "bg-blue-100 text-blue-800 border-blue-200";
    return "bg-gray-100 text-gray-800 border-gray-200";
  }, [status, loading]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-indigo-700 px-6 py-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between">
              <h1 className="text-2xl font-bold text-white">
                ONNX Double Vector Model
              </h1>
              {device && (
                <div className="mt-2 sm:mt-0 px-3 py-1 rounded-full bg-white/20 text-white text-sm font-medium">
                  Running on {device.toUpperCase()}
                </div>
              )}
            </div>
          </div>

          {/* Status bar */}
          <div className="px-6 py-3 bg-slate-50 border-b border-slate-200">
            <div className="flex items-center">
              <span className="text-sm font-medium text-slate-600 mr-2">
                Status:
              </span>
              <span
                className={`text-sm px-2.5 py-0.5 rounded-full border ${getStatusBadgeColor()}`}
              >
                {status}
              </span>

              {loading && (
                <div className="ml-2">
                  <div className="animate-spin w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
                </div>
              )}
            </div>
          </div>

          {/* Main content */}
          <div className="p-6">
            {/* Input section */}
            <div className="mb-6">
              <h2 className="text-lg font-semibold text-slate-800 mb-3">
                Input Vector
              </h2>
              <div className="flex flex-col sm:flex-row gap-3">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Enter comma-separated numbers"
                  className="flex-1 p-2 border border-slate-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                  disabled={loading}
                />
                <button
                  onClick={handleRunInference}
                  disabled={loading}
                  className={`px-5 py-2 rounded-md font-medium transition-colors ${
                    loading
                      ? "bg-slate-400 text-white cursor-not-allowed"
                      : "bg-blue-600 text-white hover:bg-blue-700 shadow-sm"
                  }`}
                >
                  {loading ? "Running..." : "Run Model"}
                </button>
              </div>
            </div>

            {/* Output section */}
            {result && (
              <div className="mb-6 p-4 border border-slate-200 rounded-lg bg-slate-50">
                <h2 className="text-lg font-semibold text-slate-800 mb-2">
                  Output Vector
                </h2>
                <div className="p-3 bg-white border border-slate-200 rounded-md font-mono text-slate-700">
                  {result.join(", ")}
                </div>
                {inferenceTime && (
                  <div className="mt-3 flex items-center text-sm text-slate-500">
                    Inference time: {inferenceTime.toFixed(2)} ms
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default OnnxDoubleModel;

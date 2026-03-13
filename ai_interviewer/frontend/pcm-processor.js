/**
 * PCM AudioWorklet Processor
 *
 * Captures raw Float32 audio samples from the microphone and forwards
 * them to the main thread for downsampling and conversion to Int16 PCM.
 *
 * This runs in the AudioWorklet thread for low-latency, glitch-free capture.
 * Reference: Google's official gemini-live-api-examples/pcm-processor.js
 */
class PCMProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input.length > 0) {
      // Send a copy of the Float32 channel data to the main thread
      const channelData = input[0];
      this.port.postMessage(new Float32Array(channelData));
    }
    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);

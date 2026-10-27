export type Gpu = {
  device: GPUDevice;
  format: GPUTextureFormat;
  adapterName: string;
  hasTimestamps: boolean;
};

export class WebGpuUnavailable extends Error {}

export async function requestGpu(): Promise<Gpu> {
  if (!navigator.gpu) throw new WebGpuUnavailable('This browser has no WebGPU.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new WebGpuUnavailable('No WebGPU adapter was found.');
  const hasTimestamps = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({ requiredFeatures: hasTimestamps ? ['timestamp-query'] : [] });
  const info = adapter.info;
  return {
    device,
    format: navigator.gpu.getPreferredCanvasFormat(),
    adapterName: info.description || [info.vendor, info.architecture].filter(Boolean).join(' ') || 'unknown GPU',
    hasTimestamps,
  };
}

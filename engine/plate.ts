// The user's own footage behind the fire: a video or a still, read from a file and kept on this device.
export type Plate = { name: string; width: number; height: number; source: HTMLVideoElement | ImageBitmap };

const MAX_SIDE = 8192;

export async function loadPlate(file: File): Promise<Plate> {
  const plate = file.type.startsWith('image/') ? await loadStill(file) : await loadVideo(file);
  if (Math.max(plate.width, plate.height) > MAX_SIDE) {
    releasePlate(plate);
    throw new Error(`Footage larger than ${MAX_SIDE} pixels a side is not supported.`);
  }
  return plate;
}

async function loadStill(file: File): Promise<Plate> {
  const source = await createImageBitmap(file).catch(() => {
    throw new Error('This browser cannot read that image.');
  });
  return { name: file.name, width: source.width, height: source.height, source };
}

async function loadVideo(file: File): Promise<Plate> {
  const source = document.createElement('video');
  source.src = URL.createObjectURL(file);
  source.muted = true;
  source.loop = true;
  source.playsInline = true;
  try {
    await source.play();
  } catch {
    URL.revokeObjectURL(source.src);
    throw new Error('This browser cannot play that file.');
  }
  return { name: file.name, width: source.videoWidth, height: source.videoHeight, source };
}

export function releasePlate({ source }: Plate) {
  if (source instanceof ImageBitmap) return source.close();
  source.pause();
  URL.revokeObjectURL(source.src);
  source.removeAttribute('src');
  source.load();
}

// Puts a video plate on the frame of a clip time, wrapping past its end as the live view loops it. Resolves once
// that frame is decoded; a still has nothing to seek.
export function seekPlate({ source }: Plate, seconds: number) {
  if (source instanceof ImageBitmap) return Promise.resolve();
  const time = Number.isFinite(source.duration) ? seconds % source.duration : seconds;
  if (Math.abs(source.currentTime - time) < 1e-4 && !source.seeking) return Promise.resolve();
  return new Promise<void>((resolve, reject) => {
    const done = () => {
      source.removeEventListener('seeked', done);
      source.removeEventListener('error', done);
      if (source.error) reject(new Error('The footage stopped decoding.'));
      else resolve();
    };
    source.addEventListener('seeked', done);
    source.addEventListener('error', done);
    source.currentTime = time;
  });
}

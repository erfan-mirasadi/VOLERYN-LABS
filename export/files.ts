import { type Bytes, crc32 } from './png';

// A zip without zip64 stops at 4 GB and 65,535 entries; the cap stays well under that, and the parts sit in memory
// until the download starts.
const ZIP_LIMIT = 2 ** 31;
const ZIP_MAX_ENTRIES = 65535;

export type Destination = {
  name: string;
  write(file: string, bytes: Bytes): Promise<void>;
  openStream?(file: string): Promise<FileSystemWritableFileStream>;
  finish(): Promise<void>;
  // Drops what a cancelled or failed job wrote, so a half-written folder never passes for an export.
  discard(): Promise<void>;
};

type DirectoryPicker = (options: { id: string; mode: 'readwrite' }) => Promise<FileSystemDirectoryHandle>;

// A subfolder of a directory the user picks (Chrome and Edge), otherwise a zip download; null when the picker is dismissed.
export async function openDestination(folder: string): Promise<Destination | null> {
  const picker = window as Window & { showDirectoryPicker?: DirectoryPicker };
  if (!picker.showDirectoryPicker) return zipDestination(folder);
  try {
    const root = await picker.showDirectoryPicker({ id: 'voleryn-export', mode: 'readwrite' });
    return directoryDestination(root, await root.getDirectoryHandle(folder, { create: true }), folder);
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') return null;
    throw error;
  }
}

function directoryDestination(root: FileSystemDirectoryHandle, dir: FileSystemDirectoryHandle, name: string): Destination {
  const openStream = async (file: string) => (await dir.getFileHandle(file, { create: true })).createWritable();
  return {
    name,
    openStream,
    async write(file, bytes) {
      const stream = await openStream(file);
      await stream.write(bytes);
      await stream.close();
    },
    finish: async () => {},
    discard: () => root.removeEntry(name, { recursive: true }).catch(() => {}),
  };
}

// Files are stored, not deflated: PNG, EXR and MP4 are already compressed.
function zipDestination(name: string): Destination {
  const parts: Bytes[] = [];
  const entries: { name: Uint8Array; crc: number; size: number; offset: number }[] = [];
  let offset = 0;
  const now = new Date();
  const time = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
  const date = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  return {
    name,
    async write(file, bytes) {
      const fileName = new TextEncoder().encode(file);
      if (offset + bytes.length > ZIP_LIMIT || entries.length === ZIP_MAX_ENTRIES) {
        throw new Error('Too big for a zip download here: pick a folder in Chrome or Edge, or shorten the clip.');
      }
      const crc = crc32(bytes);
      const header = record([[4, 0x04034b50], [2, 20], [2, 0x0800], [2, 0], [2, time], [2, date], [4, crc], [4, bytes.length], [4, bytes.length], [2, fileName.length], [2, 0]], fileName);
      entries.push({ name: fileName, crc, size: bytes.length, offset });
      parts.push(header, bytes);
      offset += header.length + bytes.length;
    },
    async finish() {
      const central = entries.map((entry) =>
        record(
          [[4, 0x02014b50], [2, 20], [2, 20], [2, 0x0800], [2, 0], [2, time], [2, date], [4, entry.crc], [4, entry.size], [4, entry.size], [2, entry.name.length], [2, 0], [2, 0], [2, 0], [2, 0], [4, 0], [4, entry.offset]],
          entry.name,
        ),
      );
      const centralSize = central.reduce((n, part) => n + part.length, 0);
      const end = record([[4, 0x06054b50], [2, 0], [2, 0], [2, entries.length], [2, entries.length], [4, centralSize], [4, offset], [2, 0]], new Uint8Array(0));
      const url = URL.createObjectURL(new Blob([...parts, ...central, end], { type: 'application/zip' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `${name}.zip`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    },
    async discard() {
      parts.length = 0;
    },
  };
}

// Little-endian fields of 2 or 4 bytes, then a name.
function record(fields: [size: 2 | 4, value: number][], name: Uint8Array): Bytes {
  const out = new Uint8Array(fields.reduce((n, [size]) => n + size, 0) + name.length);
  const view = new DataView(out.buffer);
  let at = 0;
  for (const [size, value] of fields) {
    if (size === 2) view.setUint16(at, value, true);
    else view.setUint32(at, value >>> 0, true);
    at += size;
  }
  out.set(name, at);
  return out;
}

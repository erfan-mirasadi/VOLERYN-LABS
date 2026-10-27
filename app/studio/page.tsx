import type { Metadata } from 'next';
import Studio from '@/studio/Studio';

export const metadata: Metadata = { title: 'Studio' };

export default function StudioPage() {
  return <Studio />;
}

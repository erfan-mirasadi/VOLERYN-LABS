import Link from 'next/link';
import Showcase from './showcase/Showcase';
import './landing.css';

export default function Home() {
  return (
    <main className="showcase">
      <Showcase />
      <section className="showcase-footer" aria-label="Voleryn Studio">
        <div>
          <h2>Make it yours.</h2>
          <p>Direct fire and smoke in 3D. Export layers for your shot.</p>
        </div>
        <Link prefetch={false} href="/studio">Open Studio <span aria-hidden="true">↗</span></Link>
      </section>
    </main>
  );
}

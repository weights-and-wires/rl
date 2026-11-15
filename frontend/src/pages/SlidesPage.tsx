import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { api, ApiError, type Deck, type Project } from "../lib/api";

const WIDTH = 1920;
const HEIGHT = 1080;

/**
 * Applied under every deck's own CSS. A slide is one <section> on a fixed
 * 1920 by 1080 canvas, with its padding as the margins, so the canvas is
 * border-box and elements carry no default margins; spacing comes from `gap`.
 */
const BASE_CSS = `
*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body { width: ${WIDTH}px; height: ${HEIGHT}px; overflow: hidden; background: transparent; }
body > section { position: relative; width: ${WIDTH}px; height: ${HEIGHT}px; overflow: hidden; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 18px 20px; border-bottom: 2px solid #d9ddd1; text-align: left; }
th { font-weight: 600; }
img, svg { display: block; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
`;

/** The slide's first heading, as plain text, for the slide list and labels. */
function slideTitle(html: string): string {
  const doc = new DOMParser().parseFromString(html, "text/html");
  const heading = doc.querySelector("h1, h2, .big");
  return heading?.textContent?.replace(/\s+/g, " ").trim() ?? "";
}

/**
 * A project's slide deck: one slide at a time, scaled to fit, with its
 * speaker notes underneath. Each slide renders in a sandboxed frame without
 * scripts, so the deck's CSS never touches the site's and nothing in a slide
 * can run.
 */
export default function SlidesPage() {
  const { slug = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const [deck, setDeck] = useState<Deck | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [error, setError] = useState<string | null>(null);
  const stage = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ width: 0, height: 0 });

  useEffect(() => {
    let live = true;
    setDeck(null);
    setError(null);
    api
      .slides(slug)
      .then((body) => live && setDeck(body))
      .catch((exc) => {
        if (!live) return;
        setError(
          exc instanceof ApiError && exc.status === 404
            ? "This project has no slides."
            : "Could not load the slides.",
        );
      });
    api
      .project(slug)
      .then((body) => live && setProject(body))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [slug]);

  const count = deck?.slides.length ?? 0;
  const requested = Number.parseInt(params.get("slide") ?? "1", 10);
  const index = count ? Math.min(Math.max(Number.isFinite(requested) ? requested : 1, 1), count) - 1 : 0;
  const slide = deck?.slides[index];
  const titles = useMemo(() => deck?.slides.map((s) => slideTitle(s.html)) ?? [], [deck]);

  const go = useCallback(
    (target: number) => {
      if (!count) return;
      const next = Math.min(Math.max(target, 0), count - 1);
      setParams(next === 0 ? {} : { slide: String(next + 1) }, { replace: true });
    },
    [count, setParams],
  );

  // Fit the canvas to the stage, which is 16:9 in the page and the whole
  // screen in full-screen mode.
  useEffect(() => {
    const node = stage.current;
    if (!node) return;
    const measure = () => setBox({ width: node.clientWidth, height: node.clientHeight });
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    measure();
    return () => observer.disconnect();
  }, [deck]);

  // Arrow keys, Page Up and Page Down, space, Home, and End move through the
  // deck when the keyboard is not busy in a field.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))) return;
      const moves: Record<string, number> = {
        ArrowRight: index + 1,
        ArrowDown: index + 1,
        PageDown: index + 1,
        " ": index + 1,
        ArrowLeft: index - 1,
        ArrowUp: index - 1,
        PageUp: index - 1,
        Home: 0,
        End: count - 1,
      };
      if (!(event.key in moves)) return;
      event.preventDefault();
      go(moves[event.key]);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, count, go]);

  if (error) {
    return (
      <div className="flex flex-1 items-center justify-center p-8 text-center">
        <div>
          <p className="font-serif text-lg text-ink">{error}</p>
          <Link to={`/projects/${slug}`} className="button mt-4">
            Back to the project
          </Link>
        </div>
      </div>
    );
  }

  if (!deck || !slide) {
    return (
      <div className="flex flex-1 items-center justify-center text-[13px] text-ink-4" aria-busy="true">
        Loading the slides…
      </div>
    );
  }

  const scale = box.width && box.height ? Math.min(box.width / WIDTH, box.height / HEIGHT) : 0;
  const page = `<!doctype html><html lang="en"><head><meta charset="utf-8"><style>${BASE_CSS}${deck.css}</style></head><body>${slide.html}</body></html>`;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
      <div className="mx-auto max-w-[1180px] px-4 pb-16 pt-6 sm:px-8 lg:pt-8">
        <nav aria-label="Breadcrumb" className="breadcrumbs mb-4!">
          <Link to="/">Contents</Link>
          <span aria-hidden>/</span>
          <Link to="/projects">Projects</Link>
          <span aria-hidden>/</span>
          <Link to={`/projects/${slug}`}>{project?.title ?? "Project"}</Link>
        </nav>

        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div className="min-w-0">
            <span className="kicker">Slides</span>
            <h1 className="mt-1.5 font-serif text-[30px] font-medium leading-tight tracking-[-0.03em] text-ink">
              {deck.title}
            </h1>
          </div>
          <div className="flex items-center gap-1.5">
            <button type="button" onClick={() => go(index - 1)} disabled={index === 0} className="button min-h-9! px-3!" aria-label="Previous slide">
              ←
            </button>
            <span className="meta min-w-[64px] text-center tabular-nums" aria-live="polite">
              {index + 1} / {count}
            </span>
            <button type="button" onClick={() => go(index + 1)} disabled={index === count - 1} className="button min-h-9! px-3!" aria-label="Next slide">
              →
            </button>
            <button
              type="button"
              onClick={() => stage.current?.requestFullscreen?.().catch(() => undefined)}
              className="button min-h-9! px-3!"
              title="Show the slides full screen. Press Esc to leave."
            >
              Full screen
            </button>
          </div>
        </div>

        <div
          ref={stage}
          className="slide-stage relative aspect-video w-full overflow-hidden rounded-[7px] border border-line bg-surface-muted"
          onClick={(event) => {
            // In full screen, a click on the right half moves on and the left half goes back.
            if (!document.fullscreenElement) return;
            const { left, width } = event.currentTarget.getBoundingClientRect();
            go(event.clientX - left > width / 2 ? index + 1 : index - 1);
          }}
        >
          {scale > 0 && (
            <iframe
              key={slide.id}
              title={`Slide ${index + 1}: ${titles[index] || slide.id}`}
              srcDoc={page}
              // No allow-scripts, so nothing in a slide can run. Same origin keeps
              // the frame in the page's process, so a slide paints at once.
              sandbox="allow-same-origin"
              tabIndex={-1}
              className="pointer-events-none absolute border-0"
              style={{
                width: WIDTH,
                height: HEIGHT,
                left: (box.width - WIDTH * scale) / 2,
                top: (box.height - HEIGHT * scale) / 2,
                transform: `scale(${scale})`,
                transformOrigin: "top left",
              }}
            />
          )}
        </div>

        {slide.notes && (
          <section className="mt-5 rounded-[7px] border border-line bg-surface p-5" aria-labelledby="speaker-notes">
            <h2 id="speaker-notes" className="kicker">
              Speaker notes
            </h2>
            <p className="mt-2 text-[15px] leading-[1.75] text-ink">{slide.notes}</p>
          </section>
        )}

        <ol className="mt-5 grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3" aria-label="All slides">
          {deck.slides.map((item, i) => (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => go(i)}
                aria-current={i === index ? "true" : undefined}
                className={`sidebar-lesson w-full text-left ${i === index ? "is-active" : ""}`}
              >
                <span className="sidebar-lesson-number tabular-nums">{i + 1}</span>
                <span className="min-w-0 flex-1 truncate">{titles[i] || item.id}</span>
              </button>
            </li>
          ))}
        </ol>

        <p className="meta mt-4">
          Use the arrow keys to move between slides. In full screen, click the right half to move on and the left half to go back.
        </p>
      </div>
    </div>
  );
}

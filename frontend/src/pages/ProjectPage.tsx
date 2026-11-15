import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";

import Markdown from "../components/Markdown";
import { projectMonth } from "../components/ProjectCard";
import { api, ApiError, type LessonMeta, type Project } from "../lib/api";
import { titleOf } from "../lib/parts";

type Props = {
  /** Every lesson, so the "Builds on" list shows real titles. */
  lessons: LessonMeta[];
};

/** One project's write-up, with a way into its slides and the lessons it uses. */
export default function ProjectPage({ lessons }: Props) {
  const { slug = "" } = useParams();
  const { hash } = useLocation();
  const scroller = useRef<HTMLElement>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setProject(null);
    setError(null);
    api
      .project(slug)
      .then((body) => live && setProject(body))
      .catch((exc) => {
        if (!live) return;
        setError(
          exc instanceof ApiError && exc.status === 404
            ? "There is no project at that address."
            : "Could not load the project.",
        );
      });
    return () => {
      live = false;
    };
  }, [slug]);

  // A link such as /projects/slug#s-results opens at that section.
  useEffect(() => {
    if (!hash || !project) return;
    const id = decodeURIComponent(hash.slice(1));
    const open = window.setTimeout(() => {
      scroller.current?.querySelector<HTMLElement>(`#${CSS.escape(id)}`)?.scrollIntoView({ block: "start" });
    }, 200);
    return () => window.clearTimeout(open);
  }, [hash, project]);

  if (error) {
    return (
      <div className="flex flex-1 items-center justify-center p-8 text-center">
        <div>
          <p className="font-serif text-lg text-ink">{error}</p>
          <Link to="/projects" className="button mt-4">
            Back to the projects
          </Link>
        </div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="flex flex-1 items-center justify-center text-[13px] text-ink-4" aria-busy="true">
        Loading the project…
      </div>
    );
  }

  return (
    <article ref={scroller} className="reader min-h-0 flex-1 overflow-y-auto overscroll-contain bg-paper">
      <div className="mx-auto max-w-[calc(var(--measure)+4rem)] px-5 pb-16 pt-8 sm:px-8 lg:pt-11">
        <nav aria-label="Breadcrumb" className="breadcrumbs mb-5!">
          <Link to="/">Contents</Link>
          <span aria-hidden>/</span>
          <Link to="/projects">Projects</Link>
        </nav>

        <span className="kicker">Project</span>
        <h1 className="mt-3 font-serif text-[clamp(32px,3.4vw,48px)] font-medium leading-[1.1] tracking-[-0.035em] text-ink text-balance">
          {project.title}
        </h1>
        {project.summary && <p className="mt-4 text-[16px] leading-[1.8] text-muted">{project.summary}</p>}
        <div className="lesson-meta mt-5">
          {project.date && <span>{projectMonth(project.date)}</span>}
          {project.status && <span>{project.status}</span>}
          {project.slides && <span>{project.slides.count} slides</span>}
          {project.tags.map((tag) => (
            <span key={tag}>{tag}</span>
          ))}
        </div>

        {project.slides && (
          <section className="mt-7 flex flex-col gap-4 rounded-[7px] border border-strong-line bg-surface p-6 sm:flex-row sm:items-center sm:gap-6">
            <div className="min-w-0 flex-1">
              <span className="kicker">The short version</span>
              <p className="mt-1.5 font-serif text-[24px] font-medium leading-tight tracking-[-0.025em] text-ink">
                {project.slides.title}
              </p>
              <p className="mt-1.5 text-[14px] leading-relaxed text-muted">
                {project.slides.count} slides with speaker notes. Read them first for the story, then
                this page for the details.
              </p>
            </div>
            <Link to={`/projects/${project.slug}/slides`} className="button button-primary self-start sm:self-center">
              Open the slides
              <span aria-hidden>→</span>
            </Link>
          </section>
        )}

        {(project.lessons.length > 0 || project.links.length > 0) && (
          <section className="objectives mt-7 mb-0!">
            {project.lessons.length > 0 && (
              <>
                <h2 className="objectives-title">Builds on these lessons</h2>
                <ul className="list-disc space-y-1">
                  {project.lessons.map((lesson) => (
                    <li key={lesson}>
                      <Link to={`/lessons/${lesson}`} className="text-forest underline decoration-1 underline-offset-[3px]">
                        {titleOf(lesson, lessons)}
                      </Link>
                    </li>
                  ))}
                </ul>
              </>
            )}
            {project.links.length > 0 && (
              <p className={`text-[13.5px] text-muted ${project.lessons.length > 0 ? "mt-3.5 border-t border-line pt-3" : ""}`}>
                {project.links.map((link, index) => (
                  <span key={link.url}>
                    {index > 0 && " · "}
                    <a href={link.url} className="text-forest underline decoration-1 underline-offset-[3px]" rel="noreferrer">
                      {link.label}
                    </a>
                  </span>
                ))}
              </p>
            )}
          </section>
        )}

        <div className="mt-9">
          <Markdown>{project.body}</Markdown>
        </div>

        <nav aria-label="Other pages" className="lesson-nav mt-12!">
          <Link to="/projects" rel="prev">
            <small>Back</small>
            All projects
          </Link>
          {project.slides && (
            <Link to={`/projects/${project.slug}/slides`} rel="next">
              <small>Slides</small>
              {project.slides.title}
            </Link>
          )}
        </nav>
      </div>
    </article>
  );
}

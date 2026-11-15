import { Link } from "react-router-dom";

import type { ProjectMeta } from "../lib/api";

/** "2026-11-15" → "November 2026". Anything else is shown as written. */
export function projectMonth(date: string): string {
  const match = /^(\d{4})-(\d{2})/.exec(date);
  if (!match) return date;
  const month = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, 1));
  return month.toLocaleDateString("en-US", { month: "long", year: "numeric", timeZone: "UTC" });
}

/** One project as a card, in the style of the overview's module cards. */
export default function ProjectCard({ project, index }: { project: ProjectMeta; index: number }) {
  return (
    <article className="module-card">
      <div className="module-card-header">
        <span className="module-number">
          {[projectMonth(project.date), project.status].filter(Boolean).join(" · ")}
        </span>
        <span className="module-icon" aria-hidden>
          P{index + 1}
        </span>
      </div>
      <h3>
        <Link to={`/projects/${project.slug}`} className="hover:text-forest">
          {project.title}
        </Link>
      </h3>
      <p>{project.summary}</p>
      {project.tags.length > 0 && (
        <ul className="-mt-2 mb-4 flex flex-wrap gap-1.5" aria-label="Topics">
          {project.tags.map((tag) => (
            <li key={tag} className="chip">
              {tag}
            </li>
          ))}
        </ul>
      )}
      <div className="module-card-footer">
        <span className="module-count">
          {project.slides ? `Write-up and ${project.slides.count} slides` : "Write-up"}
        </span>
        <span className="flex gap-1.5">
          {project.slides && (
            <Link
              to={`/projects/${project.slug}/slides`}
              className="button min-h-8! px-3! py-1! text-[12px]!"
              aria-label={`Slides for ${project.title}`}
            >
              Slides
            </Link>
          )}
          <Link
            to={`/projects/${project.slug}`}
            className="button min-h-8! px-3! py-1! text-[12px]!"
            aria-label={`Read ${project.title}`}
          >
            Read
            <span aria-hidden>→</span>
          </Link>
        </span>
      </div>
    </article>
  );
}

import { Link } from "react-router-dom";

import ProjectCard from "../components/ProjectCard";
import type { ProjectMeta } from "../lib/api";

type Props = {
  projects: ProjectMeta[];
  /** False until the project list has loaded, so an empty list isn't shown as "none yet". */
  loaded: boolean;
};

/**
 * The projects index: every hands-on project, newest first. A project is a
 * larger build that puts several lessons to work on a real problem, written up
 * so you can study it later.
 */
export default function ProjectsPage({ projects, loaded }: Props) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
      <div className="mx-auto max-w-[1320px] px-4 pb-16 pt-8 sm:px-8 lg:px-12 lg:pt-11">
        <nav aria-label="Breadcrumb" className="breadcrumbs">
          <Link to="/">Contents</Link>
          <span aria-hidden>/</span>
          <span>Projects</span>
        </nav>

        <section className="max-w-[720px] pb-4">
          <span className="kicker mb-4">Weights and wires · Projects</span>
          <h1 className="font-serif text-[clamp(34px,4vw,56px)] font-medium leading-[1.08] tracking-[-0.035em] text-ink">
            Projects: <em className="not-italic text-forest">the lessons at full size</em>
          </h1>
          <p className="mt-4 text-[16px] leading-[1.8] text-muted">
            Each project trains a real model on a real task with the methods from the course, then
            writes up what happened: the setup, the runs, the numbers, and the lessons it builds on.
            Read one after the lessons it links to, or open its slides for the short version.
          </p>
        </section>

        <div className="section-heading">
          <div>
            <span className="kicker">All projects</span>
            <h2 className="mt-1.5">Newest first</h2>
          </div>
        </div>

        {projects.length > 0 ? (
          <div className="module-grid">
            {projects.map((project, index) => (
              <ProjectCard key={project.slug} project={project} index={projects.length - 1 - index} />
            ))}
          </div>
        ) : (
          <p className="text-[14px] text-muted" aria-busy={!loaded}>
            {loaded ? "No projects yet." : "Loading the projects…"}
          </p>
        )}
      </div>
    </div>
  );
}

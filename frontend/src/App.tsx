import { useCallback, useEffect, useMemo, useState } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";

import AppSidebar, {
  type SidebarNavItem,
  type SidebarSection,
  type SidebarUtility,
} from "./components/AppSidebar";
import { useBook } from "./components/DownloadBook";
import Mark from "./components/Mark";
import SearchDialog from "./components/SearchDialog";
import StatusMark from "./components/StatusMark";
import AgentsPage from "./pages/AgentsPage";
import Dashboard from "./pages/Dashboard";
import LessonPage from "./pages/LessonPage";
import ProjectPage from "./pages/ProjectPage";
import ProjectsPage from "./pages/ProjectsPage";
import SlidesPage from "./pages/SlidesPage";
import { api, type Part, type ProgressMap, type ProjectMeta } from "./lib/api";
import { lessonNumbers, splitPart } from "./lib/parts";
import { partNumber, searchEntries, solvedCount } from "./lib/shellData";
import { useTheme } from "./lib/theme";

const COLLAPSED_KEY = "rl.sidebar.collapsed";

function rememberedCollapse(): boolean {
  try {
    return window.localStorage.getItem(COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

export default function App() {
  const [username, setUsername] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [parts, setParts] = useState<Part[]>([]);
  const [progress, setProgress] = useState<ProgressMap>({});
  const [projects, setProjects] = useState<ProjectMeta[]>([]);
  const [projectsLoaded, setProjectsLoaded] = useState(false);
  const [gpu, setGpu] = useState(false);
  // menuOpen drives the drawer on a narrow screen; collapsed folds the
  // sidebar to its icon rail on a wide one.
  const [menuOpen, setMenuOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(rememberedCollapse);
  const [searchOpen, setSearchOpen] = useState(false);
  const location = useLocation();
  const { mode, setMode } = useTheme();
  const book = useBook(!!username);

  const lessons = useMemo(() => parts.flatMap((part) => part.lessons), [parts]);
  const entries = useMemo(() => searchEntries(parts, projects), [parts, projects]);

  useEffect(() => {
    api
      .session()
      .then((body) => setUsername(body.username))
      .catch(() => setUsername(null))
      .finally(() => setReady(true));
  }, []);

  const refresh = useCallback(() => {
    if (!username) return;
    api
      .lessons()
      .then((body) => {
        setParts(body.parts);
        setProgress(body.progress);
      })
      .catch(() => undefined);
  }, [username]);

  useEffect(() => {
    refresh();
    if (username) api.health().then((body) => setGpu(body.gpu)).catch(() => undefined);
  }, [username, refresh]);

  useEffect(() => {
    if (!username) return;
    api
      .projects()
      .then((body) => setProjects(body.projects))
      .catch(() => undefined)
      .finally(() => setProjectsLoaded(true));
  }, [username]);

  useEffect(() => setMenuOpen(false), [location.pathname]);

  useEffect(() => {
    try {
      window.localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      // Storage being off only costs the reader their sidebar state.
    }
  }, [collapsed]);

  // Cmd or Ctrl with B folds the sidebar to its rail on a wide screen and
  // opens or closes the drawer on a narrow one. Monaco leaves it alone.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (!(event.metaKey || event.ctrlKey) || event.altKey) return;
      if (event.key !== "b" && event.key !== "B") return;
      event.preventDefault();
      if (window.matchMedia("(min-width: 1024px)").matches) setCollapsed((v) => !v);
      else setMenuOpen((v) => !v);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // "/" opens the search when the keyboard is not busy in a field or the
  // editor. Cmd or Ctrl with K opens it from anywhere.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const typing =
        !!target &&
        (target.isContentEditable ||
          ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) ||
          !!target.closest(".monaco-editor"));
      const slash = event.key === "/" && !event.metaKey && !event.ctrlKey && !event.altKey && !typing;
      const commandK = (event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === "k";
      if (!slash && !commandK) return;
      event.preventDefault();
      setMenuOpen(false);
      setSearchOpen(true);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const currentSlug = location.pathname.startsWith("/lessons/") ? location.pathname.slice("/lessons/".length) : null;
  const onSlides = /^\/projects\/[^/]+\/slides\/?$/.test(location.pathname);
  const onAgents = location.pathname === "/agents";
  const onProjects = !onSlides && (location.pathname === "/projects" || location.pathname.startsWith("/projects/"));

  const sections = useMemo<SidebarSection[]>(() => {
    const numbers = lessonNumbers(lessons);
    return parts.map((part, index) => ({
      id: part.part,
      number: partNumber(part.part, index),
      title: splitPart(part.part).name,
      lessons: part.lessons.map((lesson) => {
        const state = progress[lesson.slug]?.status;
        return {
          id: lesson.slug,
          to: `/lessons/${lesson.slug}`,
          title: lesson.title,
          number: numbers.get(lesson.slug),
          done: state === "solved",
          current: lesson.slug === currentSlug,
          mark: <StatusMark state={state} />,
        };
      }),
    }));
  }, [parts, lessons, progress, currentSlug]);

  // Projects list newest first. Slides open the newest deck, or the projects
  // page, which links every deck, when there is more than one.
  const decks = projects.filter((project) => project.slides);
  const nav: SidebarNavItem[] = [
    { to: "/", label: "Course overview", icon: "grid", active: location.pathname === "/" },
    { to: "/projects", label: "Projects", icon: "flask", count: projects.length || undefined, active: onProjects },
    ...(decks.length > 0
      ? [
          {
            to: decks.length === 1 ? `/projects/${decks[0].slug}/slides` : "/projects",
            label: "Slides",
            icon: "play" as const,
            count: decks.length,
            active: onSlides,
          },
        ]
      : []),
    { to: "/agents", label: "Agent access", icon: "key", active: onAgents },
  ];

  const utilities = useMemo<SidebarUtility[]>(
    () =>
      book
        ? [
            {
              id: "book",
              label: "Download book",
              detail: `PDF · ${book.pages} pages · ${(book.bytes / 1e6).toFixed(1)} MB`,
              icon: "download",
              href: book.url,
              download: true,
            },
          ]
        : [],
    [book],
  );

  if (!ready) {
    return <div className="h-full bg-paper" aria-busy="true" />;
  }

  if (!username) {
    const returnTo = window.location.href;
    window.location.assign(`https://auth.ifkash.dev/?rd=${encodeURIComponent(returnTo)}`);
    return <div className="h-full bg-paper" aria-busy="true" />;
  }

  const solved = solvedCount(lessons, progress);

  return (
    <div className="sh-app">
      <AppSidebar
        brand={{ name: "Reinforcement learning", kicker: "Weights and wires", mark: <Mark size={30} /> }}
        searchLabel="Find a lesson"
        onSearch={() => {
          setMenuOpen(false);
          setSearchOpen(true);
        }}
        nav={nav}
        sections={sections}
        storageKey="rl.sidebar"
        progress={{
          done: solved,
          total: lessons.length,
          caption: solved === lessons.length && lessons.length > 0 ? "Every lesson solved." : `Saved for ${username}`,
        }}
        utilities={utilities}
        status={gpu ? { label: "GPU available", title: "A RunPod GPU endpoint is configured" } : null}
        theme={{ mode, onChange: setMode }}
        account={{ username, signOutHref: "https://auth.ifkash.dev/logout" }}
        open={menuOpen}
        onOpen={() => setMenuOpen(true)}
        onClose={() => setMenuOpen(false)}
        collapsed={collapsed}
        onToggleCollapsed={() => setCollapsed((v) => !v)}
      />

      <main className="sh-main" id="main">
        <Routes>
          <Route path="/" element={<Dashboard parts={parts} progress={progress} projects={projects} />} />
          <Route
            path="/lessons/:slug"
            element={<LessonPage gpuAvailable={gpu} lessons={lessons} onProgress={refresh} />}
          />
          <Route path="/projects" element={<ProjectsPage projects={projects} loaded={projectsLoaded} />} />
          <Route path="/projects/:slug" element={<ProjectPage lessons={lessons} />} />
          <Route path="/projects/:slug/slides" element={<SlidesPage />} />
          <Route path="/agents" element={<AgentsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <SearchDialog open={searchOpen} onClose={() => setSearchOpen(false)} entries={entries} noun="lesson" />
    </div>
  );
}

/**
 * The course mark: an agent (the dot) and an environment (the square) joined
 * by the loop of action and reward, on a rounded tile. It appears in the
 * header and on the sign-in page, so it lives in one place. The favicon in
 * index.html draws the same shape.
 */
export default function Mark({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect width="24" height="24" rx="6" fill="var(--primary)" />
      <circle cx="7" cy="12" r="2.4" fill="var(--on-primary)" />
      <rect x="14.2" y="9.6" width="4.8" height="4.8" rx="1" fill="var(--on-primary)" />
      <path
        d="M8.2 9.4C10 6.2 14 6 16.4 8.4M15.8 14.8C14 17.8 10 18 7.8 14.8"
        stroke="var(--on-primary)"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
      <path d="M14.6 7.5 16.6 8.6 16.2 6.4" stroke="var(--on-primary)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M9.4 16.6 7.6 14.6 7.2 16.8" stroke="var(--on-primary)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

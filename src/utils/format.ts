export const CATEGORY_META: Record<
  string,
  { label: string; badgeClass: string }
> = {
  'ai-llm': { label: 'AI / LLM', badgeClass: 'badge-ai-llm' },
  algorithms: { label: 'Algorithms', badgeClass: 'badge-algorithms' },
  essays: { label: 'Essays', badgeClass: 'badge-essays' },
  academic: { label: 'Academic', badgeClass: 'badge-academic' },
  japan: { label: 'Japan & Life', badgeClass: 'badge-japan' },
};

export const STATUS_META: Record<
  string,
  { label: string; badgeClass: string }
> = {
  done: { label: 'Done', badgeClass: 'badge-done' },
  'in-progress': { label: 'In Progress', badgeClass: 'badge-in-progress' },
  planned: { label: 'Planned', badgeClass: 'badge-planned' },
};

export function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
  });
}

export function estimateReadingTime(body: string): string {
  const words = body.trim().split(/\s+/).length;
  const minutes = Math.max(1, Math.round(words / 200));
  return `${minutes} min read`;
}

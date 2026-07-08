export function PageHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="mb-6">
      <p className="text-2xl md:text-3xl font-extrabold uppercase tracking-tight text-accent">
        {title}
      </p>
      <p className="text-text-muted font-light mt-1">{subtitle}</p>
    </div>
  );
}

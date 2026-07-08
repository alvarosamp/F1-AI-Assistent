export function PageHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="mb-6">
      <p className="text-3xl md:text-4xl font-extrabold uppercase tracking-tight text-[#E10600]">
        {title}
      </p>
      <p className="text-gray-400 font-light mt-1">{subtitle}</p>
    </div>
  );
}

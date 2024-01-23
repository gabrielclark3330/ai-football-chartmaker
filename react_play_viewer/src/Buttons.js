


export function EmphasizedButton({link, text, onClick}) {
  return (
      <a
        href={link}
        onClick={onClick}
        className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-400 focus:ring-0 focus:ring-offset-0"
      >
        {text}
      </a>
  );
}

export function EmphasizedButtonWithArrow({link, text, onClick}) {
  return (
      <a
        href={link}
        onClick={onClick}
        className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-400 focus:ring-0 focus:ring-offset-0"
      >
        {text}<span aria-hidden="true">→</span>
      </a>
  );
}

export function EmphasizedButtonWithBackArrow({link, text, onClick}) {
  return (
      <a
        href={link}
        onClick={onClick}
        className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-400 focus:ring-0 focus:ring-offset-0"
      >
        <span aria-hidden="true">←</span>{text}
      </a>
  );
}

export function SimpleLinkWithArrow({link, text}) {
    return (
      <a href={link} className="text-sm font-semibold leading-6 text-gray-600">
        Learn more <span aria-hidden="true">→</span>
      </a>
    );
}

export function SimpleLink({link, text}) {
    return (
      <a href={link} className="text-sm font-semibold leading-6 text-gray-600">
        {text}
      </a>
    );
}
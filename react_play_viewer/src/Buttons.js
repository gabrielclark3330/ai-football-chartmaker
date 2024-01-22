


export function EmphasizedButton({link, text}) {
  return (
      <a
        href={link}
        className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-400 focus:ring-0 focus:ring-offset-0"
      >
        {text}
      </a>
  );
}

export function EmphasizedButtonWithArrow({link, text}) {
  return (
      <a
        href={link}
        className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-400 focus:ring-0 focus:ring-offset-0"
      >
        {text}<span aria-hidden="true">→</span>
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
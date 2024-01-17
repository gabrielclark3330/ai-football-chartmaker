//bg-indigo-700 bg-sky-600 bg-sky-700
export default function CallToAction() {
  return (
    <div className="mx-auto max-w-7xl py-24 sm:px-6 sm:py-32 lg:px-8">
      <div className="relative isolate overflow-hidden px-6 pt-16 sm:rounded-xl sm:px-16 md:pt-24 lg:flex lg:gap-x-20 lg:px-24 lg:pt-0 bg-white ring-1 ring-gray-900/10">
        <div className="mx-auto max-w-md text-center lg:mx-0 lg:flex-auto lg:py-12 lg:text-left">
          <h2 className="text-3xl font-bold tracking-tight sm:text-4xl text-gray-800">
            AI play chart generator
          </h2>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Every play from your game within an hour
          </p>
          <div className="mt-10 flex items-center justify-center gap-x-6 lg:justify-start">
            <a
              href="#"
              className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
            >
              Get started
            </a>
            <a
              href="#"
              className="text-sm font-semibold leading-6 text-gray-600"
            >
              Learn more <span aria-hidden="true">→</span>
            </a>
          </div>
        </div>
        <div className="relative mt-16 h-80 lg:mt-8">
          <img
            className="absolute left-0 top-0 w-[57rem] max-w-none rounded-md bg-white/5 ring-1 ring-white/10"
            src="https://tailwindui.com/img/component-images/dark-project-app-screenshot.png"
            alt="App screenshot"
            width={1824}
            height={1080}
          />
        </div>
      </div>
    </div>
  );
}

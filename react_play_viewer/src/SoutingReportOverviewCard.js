
/* 
      <a
        href={link}
        class="flex flex-row w-60 h-32 md:w-80 md:h-44 items-center bg-white border border-gray-200 rounded-lg shadow md:flex-row hover:bg-gray-100 overflow-hidden justify-self-center"
      >
        <div className="overflow-hidden h-full w-1/2">
          <img
            className="object-cover h-full"
            src={img}
            alt=""
          />
        </div>
        <div class="flex flex-col justify-between p-4 leading-normal grow-0">
          <h5 class="mb-2 text-2xl font-bold tracking-tight text-gray-900">
            {title}
          </h5>
          <p class="mb-3 font-normal text-gray-700">
            {subtext}
          </p>
        </div>
      </a>
*/

export default function ScoutingReportCard({ img, title, subtext, link }) {
  return (
    <a class="max-w-sm bg-white border border-gray-200 rounded-lg shadow overflow-hidden  hover:bg-gray-100" href={link}>
      <div class="rounded-t-lg w-full h-40 overflow-hidden">
        <img src={img} alt="" class="w-full" />
      </div>
      <div class="p-5">
        <h5 class="mb-2 text-2xl font-bold tracking-tight text-gray-900">
          {title}
        </h5>
        <p class="mb-3 font-normal text-gray-700">
          {subtext}
        </p>
      </div>
    </a>
  );
}

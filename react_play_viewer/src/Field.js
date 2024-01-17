import "./fonts.css";


function HorizontalField() {
  let yardNumber = 1;
  let hashes = [];
  for (let i = 0; i < 100; i++) {
    // each tick represents a yard on the field
    if (i % 5 != 0) {
      hashes.push(
        <svg className="absolute" width="100%" height="100%">
          <line
            className="absolute fill-grey-900/10"
            y1="36.92%"
            y2="38.07%"
            x1={`${((10 + i) / 120) * 100}%`}
            x2={`${((10 + i) / 120) * 100}%`}
            stroke="rgb(17 24 39 / 0.1)"
            strokeWidth="2px"
            style={{ vectorEffect: "non-scaling-stroke" }}
          ></line>
          <line
            className="absolute fill-grey-900/10"
            y1="61.93%"
            y2="63.08%"
            x1={`${((10 + i) / 120) * 100}%`}
            x2={`${((10 + i) / 120) * 100}%`}
            stroke="rgb(17 24 39 / 0.1)"
            strokeWidth="2px"
            style={{ vectorEffect: "non-scaling-stroke" }}
          ></line>
        </svg>
      );
    }
  }
  let yardlines = [];
  for (let i = 2; i <= 22; i++) {
    // fb field is 120 yards and has 21 yard lines
    if (i % 2 !== 0 && i < 21) {
      if (i<11) {
        yardlines.push(
          <>
          <svg
            className="absolute fill-gray-900/10"
            style={{ top: "82%", left: `${((i * 5 * 3) / 360) * 100 + 0.2}%` }}
            height="18%"
            viewBox="0 0 100 100"
          >
          <polygon transform="translate(0, -3)" points="23,45 23,55 5,50" class="triangle"></polygon>
            <text
              x="50%"
              y="50%"
              transform="rotate(0 50 50)"
              text-anchor="middle"
              dominant-baseline="middle"
              font-family="Roboto,Barlow Condensed,sans-serif"
              fontSize="36"
            >
              {yardNumber}0
            </text>
          </svg>

          <svg
            className="absolute fill-gray-900/10"
            style={{ top: "0%", left: `${((i * 5 * 3) / 360) * 100 + 0.2}%` }}
            height="18%"
            viewBox="0 0 100 100"
          >
          <polygon transform="translate(0, 2)" points="23,45 23,55 5,50" class="triangle"></polygon>
            <text
              x="50%"
              y="50%"
              transform="rotate(180 50 50)"
              text-anchor="middle"
              dominant-baseline="middle"
              font-family="Roboto,Barlow Condensed,sans-serif"
              fontSize="36"
            >
              {yardNumber}0
            </text>
          </svg>
          </>
        );
      }else if (i>11){
        yardlines.push(
          <>
          <svg
            className="absolute fill-gray-900/10"
            style={{ top: "82%", left: `${((i * 5 * 3) / 360) * 100 + 0.2}%` }}
            height="18%"
            viewBox="0 0 100 100"
          >
            <text
              x="50%"
              y="50%"
              transform="rotate(0 50 50)"
              text-anchor="middle"
              dominant-baseline="middle"
              font-family="Roboto,Barlow Condensed,sans-serif"
              fontSize="36"
            >
              {yardNumber}0
            </text>
          <polygon transform="translate(0, -3)" points="77,45 77,55 95,50" class="triangle"></polygon>
          </svg>

          <svg
            className="absolute fill-gray-900/10"
            style={{ top: "0%", left: `${((i * 5 * 3) / 360) * 100 + 0.2}%` }}
            height="18%"
            viewBox="0 0 100 100"
          >
          <polygon transform="translate(0, 2)" points="77,45 77,55 95,50" class="triangle"></polygon>
            <text
              x="50%"
              y="50%"
              transform="rotate(180 50 50)"
              text-anchor="middle"
              dominant-baseline="middle"
              font-family="Roboto,Barlow Condensed,sans-serif"
              fontSize="36"
            >
              {yardNumber}0
            </text>
          </svg>
          </>
        );
      }else if(i==11) {
        yardlines.push(
          <>
          <svg
            className="absolute fill-gray-900/10"
            style={{ top: "82%", left: `${((i * 5 * 3) / 360) * 100 + 0.2}%` }}
            height="18%"
            viewBox="0 0 100 100"
          >
            <text
              x="50%"
              y="50%"
              transform="rotate(0 50 50)"
              text-anchor="middle"
              dominant-baseline="middle"
              font-family="Roboto,Barlow Condensed,sans-serif"
              fontSize="36"
            >
              {yardNumber}0
            </text>
          </svg>

          <svg
            className="absolute fill-gray-900/10"
            style={{ top: "0%", left: `${((i * 5 * 3) / 360) * 100 + 0.2}%` }}
            height="18%"
            viewBox="0 0 100 100"
          >
            <text
              x="50%"
              y="50%"
              transform="rotate(180 50 50)"
              text-anchor="middle"
              dominant-baseline="middle"
              font-family="Roboto,Barlow Condensed,sans-serif"
              fontSize="36"
            >
              {yardNumber}0
            </text>
          </svg>
          </>
        );

      }
      yardlines.push(
        <>
          <div
            style={{ left: `${((i * 5 * 3) / 360) * 100}%` }} // i times 3 to go from yards to feet
            className="absolute w-0.5 bg-gray-900/10 h-full"
          ></div>
        </>
      );
      if (i >= 11) {
        yardNumber -= 1;
      } else {
        yardNumber += 1;
      }
    } else {
      yardlines.push(
        <>
          <div
            style={{ left: `${((i * 5 * 3) / 360) * 100}%` }} // i times 3 to go from yards to feet
            className="absolute w-0.5 bg-gray-900/10 h-full"
          ></div>
        </>
      );
    }
  }

  return (
    <div className="m-20">
      <div className="relative w-3/4 aspect-[9/4] bg-white ring-1 ring-gray-900/10 rounded-xl">
        {yardlines}
        {hashes}
      </div>
    </div>
  );
}

function VerticalField() {
  let yardNumber = 1;
  let hashes = [];
  for (let i = 0; i < 100; i++) {
    // each tick represents a yard on the field
    if (i % 5 != 0) {
      hashes.push(
        <svg className="absolute" width="100%" height="100%">
          <line
            className="absolute fill-grey-900/10"
            x1="36.92%"
            x2="38.07%"
            y1={`${((10 + i) / 120) * 100}%`}
            y2={`${((10 + i) / 120) * 100}%`}
            stroke="rgb(17 24 39 / 0.1)"
            strokeWidth="2px"
            style={{ vectorEffect: "non-scaling-stroke" }}
          ></line>
          <line
            className="absolute fill-grey-900/10"
            x1="61.93%"
            x2="63.08%"
            y1={`${((10 + i) / 120) * 100}%`}
            y2={`${((10 + i) / 120) * 100}%`}
            stroke="rgb(17 24 39 / 0.1)"
            strokeWidth="2px"
            style={{ vectorEffect: "non-scaling-stroke" }}
          ></line>
        </svg>
      );
    }
  }
  let yardlines = [];
  for (let i = 2; i <= 22; i++) {
    // fb field is 120 yards and has 21 yard lines
    if (i % 2 !== 0 && i < 21) {
      if (i<11) {
        yardlines.push(
          <>
          <svg
            className="absolute fill-gray-900/10"
            style={{ left: "82%", top: `${((i * 5 * 3) / 360) * 100 + 0.2}%` }}
            width="18%"
            viewBox="0 0 100 100"
          >
          <polygon transform="translate(-3, 0)" points="45,23 55,23 50,5" class="triangle"></polygon>
            <text
              x="50%"
              y="50%"
              transform="rotate(-90 50 50)"
              text-anchor="middle"
              dominant-baseline="middle"
              font-family="Roboto,Barlow Condensed,sans-serif"
              fontSize="36"
            >
              {yardNumber}0
            </text>
          </svg>

          <svg
            className="absolute fill-gray-900/10"
            style={{ left: "0%", top: `${((i * 5 * 3) / 360) * 100 + 0.2}%` }}
            width="18%"
            viewBox="0 0 100 100"
          >
          <polygon transform="translate(2, 0)" points="45,23 55,23 50,5" class="triangle"></polygon>
            <text
              x="50%"
              y="50%"
              transform="rotate(90 50 50)"
              text-anchor="middle"
              dominant-baseline="middle"
              font-family="Roboto,Barlow Condensed,sans-serif"
              fontSize="36"
            >
              {yardNumber}0
            </text>
          </svg>
          </>
        );
      }else if (i>11){
        yardlines.push(
          <>
          <svg
            className="absolute fill-gray-900/10"
            style={{ left: "82%", top: `${((i * 5 * 3) / 360) * 100 + 0.2}%` }}
            width="18%"
            viewBox="0 0 100 100"
          >
            <text
              x="50%"
              y="50%"
              transform="rotate(-90 50 50)"
              text-anchor="middle"
              dominant-baseline="middle"
              font-family="Roboto,Barlow Condensed,sans-serif"
              fontSize="36"
            >
              {yardNumber}0
            </text>
          <polygon transform="translate(-3, 0)" points="45,77 55,77 50,95" class="triangle"></polygon>
          </svg>

          <svg
            className="absolute fill-gray-900/10"
            style={{ left: "0%", top: `${((i * 5 * 3) / 360) * 100 + 0.2}%` }}
            width="18%"
            viewBox="0 0 100 100"
          >
          <polygon transform="translate(2, 0)" points="45,77 55,77 50,95" class="triangle"></polygon>
            <text
              x="50%"
              y="50%"
              transform="rotate(90 50 50)"
              text-anchor="middle"
              dominant-baseline="middle"
              font-family="Roboto,Barlow Condensed,sans-serif"
              fontSize="36"
            >
              {yardNumber}0
            </text>
          </svg>
          </>
        );
      }else if(i==11) {
        yardlines.push(
          <>
          <svg
            className="absolute fill-gray-900/10"
            style={{ left: "82%", top: `${((i * 5 * 3) / 360) * 100 + 0.2}%` }}
            width="18%"
            viewBox="0 0 100 100"
          >
            <text
              x="50%"
              y="50%"
              transform="rotate(-90 50 50)"
              text-anchor="middle"
              dominant-baseline="middle"
              font-family="Roboto,Barlow Condensed,sans-serif"
              fontSize="36"
            >
              {yardNumber}0
            </text>
          </svg>

          <svg
            className="absolute fill-gray-900/10"
            style={{ left: "0%", top: `${((i * 5 * 3) / 360) * 100 + 0.2}%` }}
            width="18%"
            viewBox="0 0 100 100"
          >
            <text
              x="50%"
              y="50%"
              transform="rotate(90 50 50)"
              text-anchor="middle"
              dominant-baseline="middle"
              font-family="Roboto,Barlow Condensed,sans-serif"
              fontSize="36"
            >
              {yardNumber}0
            </text>
          </svg>
          </>
        );

      }
      yardlines.push(
        <>
          <div
            style={{ top: `${((i * 5 * 3) / 360) * 100}%` }} // i times 3 to go from yards to feet
            className="absolute h-0.5 bg-gray-900/10 w-full"
          ></div>
        </>
      );
      if (i >= 11) {
        yardNumber -= 1;
      } else {
        yardNumber += 1;
      }
    } else {
      yardlines.push(
        <>
          <div
            style={{ top: `${((i * 5 * 3) / 360) * 100}%` }} // i times 3 to go from yards to feet
            className="absolute h-0.5 bg-gray-900/10 w-full"
          ></div>
        </>
      );
    }
  }

  return (
    <div className="m-20">
      <div className="relative w-2/3 aspect-[4/9] bg-white ring-1 ring-gray-900/10 rounded-xl">
        {yardlines}
        {hashes}
      </div>
    </div>
  );
}

export {VerticalField, HorizontalField};

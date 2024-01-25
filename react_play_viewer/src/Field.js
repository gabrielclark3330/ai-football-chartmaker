import "./fonts.css";

function YardMarkings({
  yardNumber,
  isHorizontal,
  isHomeSide,
  fieldPos,
  showArrow,
  isArrowRight,
}) {
  let polygon = [];
  if (showArrow) {
    if (!isArrowRight) {
      polygon = [
        <polygon
          transform={isHomeSide ? "translate(0, -3)" : "translate(0, 2)"}
          points={isHorizontal?"23,45 23,55 5,50":"45,23 55,23 50,5"}
          class="triangle"
        />,
      ];
    }else{
      polygon = [
        <polygon
          transform={isHomeSide?isHorizontal?"translate(0, -3)":"translate(-3, 0)":isHorizontal?"translate(0, 2)":"translate(2, 0)"}
          points={isHorizontal?"77,45 77,55 95,50":"45,77 55,77 50,95"}
          class="triangle"
        />
      ];
    }
  }
    return (
      <svg
        className="absolute fill-gray-900/10"
        style={{
          top:isHorizontal?isHomeSide ? "82%" : "0%":fieldPos,
          left:isHorizontal?fieldPos:isHomeSide ? "82%" : "0%",
        }}
        height={isHorizontal?"18%":null}
        width={isHorizontal?null:"18%"}
        viewBox="0 0 100 100"
      >
        {polygon}
        <text
          x="50%"
          y="50%"
          transform={isHorizontal?isHomeSide ? "rotate(0 50 50)" : "rotate(180 50 50)":isHomeSide ? "rotate(-90 50 50)" : "rotate(90 50 50)"}
          text-anchor="middle"
          dominant-baseline="middle"
          font-family="Roboto,Barlow Condensed,sans-serif"
          fontSize="36"
        >
          {yardNumber}0
        </text>
      </svg>
    );
}

function Field({ isHorizontal, driveJson }) {
  let yardNumber = 1;
  let hashes = [];
  let drives = null;
  if (driveJson) {
    const gameID = Object.keys(driveJson).filter((key)=>!isNaN(key))[0];
    const homeTeamName=driveJson[gameID].home.abbr;
    const awayTeamName=driveJson[gameID].away.abbr;
    console.log(homeTeamName);
    console.log(awayTeamName);
    drives = driveJson[gameID].drives;
  }
  for (const driveKey of Object.keys(drives)) {
    if(!isNaN(driveKey)){
      console.log(drives[driveKey].start.yrdln);
      console.log(drives[driveKey].end.yrdln);
    }
  }
  for (let i = 0; i < 100; i++) {
    // each tick represents a yard on the field
    if (i % 5 !== 0) {
      hashes.push(
        <svg className="absolute" width="100%" height="100%">
          <line
            className="absolute fill-grey-900/10"
            y1={isHorizontal ? "36.92%" : `${((10 + i) / 120) * 100}%`}
            y2={isHorizontal ? "38.07%" : `${((10 + i) / 120) * 100}%`}
            x1={isHorizontal ? `${((10 + i) / 120) * 100}%` : "36.92%"}
            x2={isHorizontal ? `${((10 + i) / 120) * 100}%` : "38.07%"}
            stroke="rgb(17 24 39 / 0.1)"
            strokeWidth="2px"
            style={{ vectorEffect: "non-scaling-stroke" }}
          />
          <line
            className="absolute fill-grey-900/10"
            y1={isHorizontal ? "61.93%" : `${((10 + i) / 120) * 100}%`}
            y2={isHorizontal ? "63.08%" : `${((10 + i) / 120) * 100}%`}
            x1={isHorizontal ? `${((10 + i) / 120) * 100}%` : "61.93%"}
            x2={isHorizontal ? `${((10 + i) / 120) * 100}%` : "63.08%"}
            stroke="rgb(17 24 39 / 0.1)"
            strokeWidth="2px"
            style={{ vectorEffect: "non-scaling-stroke" }}
          />
        </svg>
      );
    }
  }
  let yardlines = [];
  for (let i = 2; i <= 22; i++) {
    // fb field is 120 yards and has 21 yard lines
    if (i % 2 !== 0 && i < 21) {
      if (i < 11) {
        yardlines.push(
          <>
            <YardMarkings
              yardNumber={yardNumber}
              isHorizontal={isHorizontal}
              isHomeSide={true}
              fieldPos={`${((i * 5 * 3) / 360) * 100 + 0.2}%`}
              showArrow={true}
              isArrowRight={false}
            />
            <YardMarkings
              yardNumber={yardNumber}
              isHorizontal={isHorizontal}
              isHomeSide={false}
              fieldPos={`${((i * 5 * 3) / 360) * 100 + 0.2}%`}
              showArrow={true}
              isArrowRight={false}
            />
          </>
        );
      } else if (i > 11) {
        yardlines.push(
          <>
            <YardMarkings
              yardNumber={yardNumber}
              isHorizontal={isHorizontal}
              isHomeSide={true}
              fieldPos={`${((i * 5 * 3) / 360) * 100 + 0.2}%`}
              showArrow={true}
              isArrowRight={true}
            />
            <YardMarkings
              yardNumber={yardNumber}
              isHorizontal={isHorizontal}
              isHomeSide={false}
              fieldPos={`${((i * 5 * 3) / 360) * 100 + 0.2}%`}
              showArrow={true}
              isArrowRight={true}
            />
          </>
        );
      } else if (i === 11) {
        yardlines.push(
          <>
            <YardMarkings
              yardNumber={yardNumber}
              isHorizontal={isHorizontal}
              isHomeSide={true}
              fieldPos={`${((i * 5 * 3) / 360) * 100 + 0.2}%`}
              showArrow={false}
            />
            <YardMarkings
              yardNumber={yardNumber}
              isHorizontal={isHorizontal}
              isHomeSide={false}
              fieldPos={`${((i * 5 * 3) / 360) * 100 + 0.2}%`}
              showArrow={false}
            />
          </>
        );
      }
      if (isHorizontal){
        yardlines.push(
            <div
              style={{ left: `${((i * 5 * 3) / 360) * 100}%` }} // i times 3 to go from yards to feet
              className={"absolute bg-gray-900/10 h-full w-0.5"}
            />
        );
      }else{
        yardlines.push(
            <div
              style={{ top: `${((i * 5 * 3) / 360) * 100}%` }} // i times 3 to go from yards to feet
              className={"absolute bg-gray-900/10 w-full h-0.5"}
            />
        );
      }
      if (i >= 11) {
        yardNumber -= 1;
      } else {
        yardNumber += 1;
      }
    } else {
      if (isHorizontal){
        yardlines.push(
            <div
              style={{ left: `${((i * 5 * 3) / 360) * 100}%` }} // i times 3 to go from yards to feet
              className="absolute w-0.5 bg-gray-900/10 h-full"
            />
        );
      }else{
        yardlines.push(
            <div
              style={{ top: `${((i * 5 * 3) / 360) * 100}%` }} // i times 3 to go from yards to feet
              className="absolute h-0.5 bg-gray-900/10 w-full"
            />
        );
      }
    }
  }
  if (isHorizontal){
    return (
      <div className="relative w-full aspect-[9/4] bg-white ring-1 ring-gray-900/10 rounded-xl">
        {yardlines}
        {hashes}
      </div>
    );
  }else{
    return (
      <div className="relative w-full aspect-[4/9] bg-white ring-1 ring-gray-900/10 rounded-xl">
        {yardlines}
        {hashes}
      </div>
    );
  }
}

function HorizontalField({driveJson}){
  return(Field({isHorizontal:true, driveJson:driveJson}));
}

function VerticalField({driveJson}){
  return(Field({isHorizontal:false, driveJson:driveJson}));
}

export { VerticalField, HorizontalField };

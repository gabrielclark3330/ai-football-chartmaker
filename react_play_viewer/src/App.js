import './App.css';
import { BundleCurve } from 'react-svg-curve';
import footballField from './footballFieldDiagram.png';
//import playerPaths from './SteelersExample.mp4_player_detections_json_string.json';
//import playerClasses from './SteelersExample.mp4_player_classes_json_string.json';
//import playerPaths from './CardsExample.mp4_player_detections_json_string.json';
//import playerClasses from './CardsExample.mp4_player_classes_json_string.json';
import playerPaths from './BillsExample.mp4_player_detections_json_string.json';
import playerClasses from './BillsExample.mp4_player_classes_json_string.json';

/*
diagram_info = {
    "boxes": [
        {"label": "t10l","x": 77.12,"y": 71.28,},
        {"label": "t10r","x": 513.97,"y": 66.90,},
        {"label": "t20l","x": 76.91,"y": 201.75,},
        {"label": "t20r","x": 514.18,"y": 201.34,},
        {"label": "t30l","x": 76.91,"y": 336.19,},
        {"label": "t30r","x": 514.60,"y": 336.19,},
        {"label": "t40l","x": 77.33,"y": 468.95,},
        {"label": "t40r","x": 514.18,"y": 470.41,},
        {"label": "50l","x": 76.70,"y": 606.10,},
        {"label": "50r","x": 514.60,"y": 602.76,},
        {"label": "b40l","x": 77.12,"y": 739.07,},
        {"label": "b40r","x": 514.39,"y": 739.07,},
        {"label": "b30l","x": 76.70,"y": 873.92,},
        {"label": "b30r","x": 513.97,"y": 873.71,},
        {"label": "b20l","x": 76.91,"y": 1009.19,},
        {"label": "b20r","x": 514.60,"y": 1008.98,},
        {"label": "b10l","x": 76.70,"y": 1144.45,},
        {"label": "b10r","x": 513.56,"y": 1141.12,}
    ],
    "width": 590,
    "height": 1215,
    "key": "footballFieldDiagram.png",
}
*/

const CircleWithLetter = ({ x, y, letter, color }) => {
  return (
    <svg style={{ position: 'absolute', left: x-10, top: y-10 }}>
      <circle cx={20} cy={20} r="15" fill={color} />
      <text x={20} y={20} fill="white" fontSize="20" textAnchor="middle" dy=".3em">{letter}</text>
    </svg>
  );
};

const PathWithEnding = ({ data, color, beta }) => {

  return (
    <svg style={{ position: 'absolute', left: 0, top: 0 }} width="590" height="1215" stroke-width="5" strokeLinecap='round'>
      <BundleCurve
        data={data}
        beta={beta}
        showPoints={false}
        stroke={color}
      />
    </svg>
  );
}

const MyImageComponent = () => {
  const offsetx = 90;
  const offsety = 0;
  const chartWidth = 590;
  console.log(playerPaths);
  console.log(playerClasses);
  const circles = [];
  const paths = [];
  //#395E66
  //"#DBD5B5"
  const classColors = ["#DBD5B5", "#4D4D4D", "#E50101", "#000000", "#00746B"]; // class_id=[defense(0), oline(1), qb(2), ref(3), skill(4)]
  const playerLetters = ["D", "OL", "Q", "R", ""]
  for (var trackid in playerClasses) {
    var playerClass = playerClasses[trackid];
    const data = [];
    playerPaths[trackid].forEach(track => {
      data.push([((chartWidth-track[0][0])+offsetx)*.75, track[0][1]+offsety]);
    });
    if (playerClass === 2 || playerClass === 4) {

      circles.push(
        <>
          <CircleWithLetter x={data[0][0]} y={data[0][1]} letter={playerLetters[playerClass]} color={classColors[playerClass]} />
        </>);
      paths.push(<>
        <PathWithEnding data={data} beta={1} color={classColors[playerClass]} />
      </>);
    } else {
      circles.push(
        <>
          <CircleWithLetter x={data[0][0]} y={data[0][1]} letter={playerLetters[playerClass]} color={classColors[playerClass]} />
        </>
      );
    }
  }
  return (
    <div style={{ position: 'relative' }}>
      <img src={footballField} alt="Football Field" width="590" height="1215" />
      {paths}
      {circles}
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <MyImageComponent />
      </header>
    </div>
  );
}

export default App;

import { HorizontalField } from "./Field";
import game1 from "./gameDriveJson/2009080950.json";
import game2 from "./gameDriveJson/2009081351.json";
import game3 from "./gameDriveJson/2009081352.json";
import game4 from "./gameDriveJson/2009081353.json";
import game5 from "./gameDriveJson/2009081450.json";

export default function DriveCharts({gameDetailsJson}) {
    const games = [game1, game2, game3, game4, game5];
    const gamesNames = ["game1", "game2", "game3", "game4", "game5"];
    return (
        <div className="flex flex-row w-full h-56 gap-4 overflow-x-auto rounded-lg shadow-lg ring-1 ring-black/5 p-4 bg-white">
            {games.map((game, index)=>{
                return(
                    <div className="flex-1 p-2" key={index}>
                        <div className="flex flex-row justify-center text-md font-medium text-gray-900">
                            {gamesNames[index]}
                        </div>
                        <div className="flex h-full p-2 hover:bg-gray-200 rounded-lg">
                            <HorizontalField driveJson={game}/>
                        </div>
                    </div>
                );})}
        </div>
    );
}
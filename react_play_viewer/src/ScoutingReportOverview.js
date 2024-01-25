import AppHeader from "./AppHeader";
import DriveCharts from "./DriveChart";
import Footer from "./Footer";
import ReportTitle from "./ReportTitle";
import Table from "./Table";
import { useParams } from "react-router-dom";

export default function ScoutingReportOverview() {
    let tables = [
        [["Run/Pass", "Count", "%", "Total Yds", "Avg Yds"],
        ["Pass", "42", "36.8%", "167", "4.3"],
        ["Run", "72", "63.2%", "374", "5.2"]],
        [["Play Direction", "Run", "Run %", "Pass", "Pass %"],
        ["R", "41", "56.2%", "28", "38.4%"],
        ["L", "31", "66.0%", "14", "29.8"]],
        [["Down", "Run", "Run %", "Pass", "Pass %"],
        ["1st", "40", "80.0%", "8", "16.0%"],
        ["2nd", "24", "61.5%", "13", "33.3%"],
        ["3rd", "7", "29.2%", "15", "62.5%"],
        ["4th", "1", "14.3%", "6", "85.7%"]],
        [["Distance", "Run", "Run %", "Pass", "Pass %"],
        ["1-3", "9", "81.8%", "1", "9.1%"],
        ["1-4", "13", "61.9%", "8", "38.1%"],
        ["7-9", "6", "23.1%", "18", "69.2%"],
        ["10+", "21", "72.4%", "8", "27.6%"]],
        [["Hash", "Run", "Run %", "Pass", "Pass %"],
        ["R", "20", "80.0%", "5", "20.0%"],
        ["M", "5", "83.3%", "1", "16.7%"],
        ["L", "11", "61.1%", "7", "38.9%"]],
        [["Yard Line", "Run", "Run %", "Pass", "Pass %"],
        ["0 to -20", "6", "85.7%", "1", "14.3%"],
        ["-20 to 50", "18", "78.3%", "5", "21.7%"],
        ["50 to 20", "12", "63.2%", "7", "36.8%"],
        ["20 to 0", "0", "-", "0", "-"]]
    ];
    const {reportID, reportDate} = useParams();
    return (
        <div className="bg-[#f5f5f7] min-h-screen flex flex-col">
            <AppHeader headerTitles={["Library", "Your Team"]} headerLinks={["/home", "#"]}/>
            <div className="flex justify-center w-full my-14">
                <ReportTitle reportID={reportID} reportDate={reportDate}/>
            </div>
            <div className="flex justify-center w-full mb-7 p-4">
                <DriveCharts/>
            </div>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(22rem,1fr))] gap-4 p-4">
                {tables.map((table, index) => {
                    return (
                        <Table tableAsArray={table} key={index}/>
                    );
                })}
            </div>
            <Footer />
        </div>
    );
}

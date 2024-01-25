import { EmphasizedButtonWithBackArrow } from "./Buttons";

export default function ReportTitle({ reportID, reportDate }) {
    return (
        <div className="flex flex-row items-center p-5 rounded-lg shadow-lg ring-1 ring-black/5 bg-white w-3/4">
            <div className="flex-1 flex justify-start">
                <EmphasizedButtonWithBackArrow text={"Back"} link={"/home"} />
            </div>
            <div className="flex-1">
                <h5 className="mb-2 text-2xl font-bold tracking-tight text-gray-900 text-center">
                    {reportID}
                </h5>
                <p className="mb-3 font-normal text-center text-gray-700">
                    {reportDate}
                </p>
            </div>
            <div className="flex-1">

            </div>
        </div>
    );
}


export default function Table({ tableAsArray }) {
    let heading = tableAsArray[0];
    let rows = tableAsArray.slice(1);
    return (
        <div class="relative overflow-x-auto rounded-lg shadow-lg ring-1 ring-black/5">
            <table class="w-full text-sm text-left text-gray-500">
                <thead class="text-xs text-gray-700 bg-gray-50">
                    <tr>
                        {heading.map((colHeading, index) => {
                            return (
                                <th scope="col" class="px-3 py-3" key={index}>
                                    {colHeading}
                                </th>
                            );
                        })}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row, index)=>{
                        return(
                    <tr class={index<rows.length-1?"bg-white border-b":"bg-white"} key={index}>
                        <th scope="row" class="px-3 py-4 font-medium text-gray-900 whitespace-nowrap">
                            {row[0]}
                        </th>
                        {row.slice(1).map((element, index)=>{
                            return(
                                <td class="px-3 py-4">
                                    {element}
                                </td>
                            );
                        })}
                    </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

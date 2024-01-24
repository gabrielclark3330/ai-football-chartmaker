import FilmUpload from "./FilmUpload";
import AppHeader from "./AppHeader";
import Footer from "./Footer";
import ScoutingReportCard from "./ScoutingReportOverviewCard";


function getRandomDate() {
  const startYear = 2017; // Start from 2017
  const endYear = 2024;   // Up to 2022
  const year = Math.floor(Math.random() * (endYear - startYear + 1)) + startYear;
  const month = Math.floor(Math.random() * 12) + 1; // Random month from 1 to 12
  const day = Math.floor(Math.random() * 28) + 1;   // Random day from 1 to 28 (assuming no leap years)

  // Format the date as "YYYY-MM-DD"
  const date = `${year}-${month.toString().padStart(2, '0')}-${day.toString().padStart(2, '0')}`;
  return date;
}

export default function HomePage() {
  let imgs = [
    "https://static.www.nfl.com/image/private/t_editorial_landscape_mobile/f_auto/league/xutbmai4wtzcndofvjc1.jpg",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThh1yca6tfHUSZJrLgtgIWI2ARlUFo-sZLjrmNIEd6r6jghHVZr9Ya1wBhZTJvfbMHYF0&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQtR0D8LY2iTINW-dDKlvD9VVvx81t9t-L-UImm4TD8uWFfptZ9TFj1MW0vS9O7dtjojBE&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYTqxHkYBLZIDvWEliE_iBxGiwAG5Xb0wY4NIUCiSFGysHlgQJ8yEXfQIL1zHTbgfjq9g&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTnsz6lpYzBSVHXIa69rOoF2ZzbZ57NM2eSOJ4K9PJ6e8wfSViYC-_LAZkZzoIj6lIUCas&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThf6BbK1KcDu5gKBVLnYioNrN4-jBBc63qqYM1mQvdGpBdCNv6jwMH9RBqRlTbIuw2_y4&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGo5PNn3Tg5CQzcsIFncch4zRCg6Kx_1WLhqNhKybrtaPh-0ZxrZQTBNDA41RDw-GK67Y&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRBQzwsjHMEn8pmDgvEBw43SbZIcvRPZHtwiprFuCveYTUoAz8zGUfhobVn5f6hUf9da8s&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQPMZP7U0go5IVE4W6n2uOecWy_z7-SbH2solgXyiBRrS-TyHkEcrLYAIv_rdUQLpVFVJk&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQtR0D8LY2iTINW-dDKlvD9VVvx81t9t-L-UImm4TD8uWFfptZ9TFj1MW0vS9O7dtjojBE&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSH5BhXZ_391SFBSMaMwpq-Z_u2gE8QR0fjeTBW257-C6k9l4nnT_uqXXQYVhxO2IONbkE&usqp=CAU"
  ];

  const nflTeams = [
    'New England Patriots',
    'Kansas City Chiefs',
    'Green Bay Packers',
    'San Francisco 49ers',
    'Baltimore Ravens',
    'Tampa Bay Buccaneers',
    'Dallas Cowboys',
    'Los Angeles Rams',
    'Buffalo Bills',
    'Chicago Bears',
    'New Orleans Saints',
    'Seattle Seahawks',
    'Pittsburgh Steelers',
    'Indianapolis Colts',
    'Miami Dolphins',
    'Minnesota Vikings'
  ];

  return (
    <div className="bg-[#f5f5f7] min-h-screen flex flex-col">
      <AppHeader headerTitles={["Library", "Your Team"]} headerLinks={["/home", "#"]}/>
      <div className="flex justify-center w-full my-14">
        <FilmUpload />
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(24rem,1fr))] gap-4">
        {imgs.map((img, index) => {
          let date = getRandomDate();
          return (
            <ScoutingReportCard
              key={index}
              link={"/report/"+nflTeams[index]+"/"+date}
              title={nflTeams[index]}
              subtext={date}
              img={img}
            />
          );
        })}
      </div>
      <Footer />
    </div>
  );
}

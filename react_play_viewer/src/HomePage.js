import FilmUpload from "./FilmUpload";
import HomePageHeader from "./HomePageHeader";
import Footer from "./Footer";

export default function HomePage() {
  return (
    <div className="bg-[#f5f5f7] min-h-screen flex flex-col">
        <HomePageHeader/>
        <div className="flex justify-center w-full my-14">
            <FilmUpload/>
        </div>
        <Footer/>
    </div>
  );
}

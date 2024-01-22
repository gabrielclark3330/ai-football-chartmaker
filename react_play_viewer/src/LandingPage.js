import Header from "./Header";
import CallToAction from "./CallToAction";
import Title from "./Title";
import FieldOld from "./FieldRendering";
import Footer from "./Footer";
import { VerticalField, HorizontalField } from "./Field";

function LandingPage() {
  return (
    <div className="bg-[#f5f5f7] min-h-screen flex flex-col">
      <Header />
      <Title />
      <CallToAction />
      <HorizontalField />
      <VerticalField />
      <FieldOld />
      <Footer />
    </div>
  );
}

export default LandingPage;

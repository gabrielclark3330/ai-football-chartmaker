import Header from "./Header";
import CallToAction from "./CallToAction";
import Title from "./Title";
import FieldOld from "./FieldRendering"
import { VerticalField, HorizontalField } from "./Field";

function App() {
  return (
    <div className="bg-[#f5f5f7]">
        <Header/>
        <Title/>
        <CallToAction/>
        <HorizontalField/>
        <VerticalField/>
        <FieldOld/>
    </div>
  );
}

export default App;

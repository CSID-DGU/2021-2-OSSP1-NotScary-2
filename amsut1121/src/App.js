import React from "react";
import { Route } from "react-router-dom";
import Header from "./components/Header";
import Main from "./components/Main";
import Pose from "./components/Pose";
import PoseHistory from "./components/PoseHistory";
import StartCheck2 from "./components/StartCheck2";

function App() {
  return (
    <>
      <Header />
      <Route path="/" exact={true} component={Main} />
      <Route path="/pose" component={Pose} />
      <Route path="/start" component={StartCheck2} />
      <Route path="/history" component={PoseHistory} />
    </>
  );
}

export default App;

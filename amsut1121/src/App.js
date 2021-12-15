import React from "react";
import { Route } from "react-router-dom";
import Header from "./components/Header";
import Pose from "./components/Pose";
import StartCheck from "./components/StartCheck";
import poseHistory from "./components/poseHistory";

function App() {
  return (
    <>
      <Header />
      <Route path="/" exact={true} component={StartCheck} />
      <Route path="/pose" component={Pose} />
      <Route path="/history" component={poseHistory} />
    </>
  );
}

export default App;

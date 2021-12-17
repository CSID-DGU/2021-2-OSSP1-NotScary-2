import React from "react";

function Main() {
  return (
    <div className="start">
      <button
        className="startButton"
        onClick={() => {
          window.location.href = "/start";
        }}
      >
        Start
      </button>
      <br />
      <button
        className="historyButton"
        onClick={() => {
          window.location.href = "/history";
        }}
      >
        History
      </button>
    </div>
  );
}

export default Main;

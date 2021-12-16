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
    </div>
  );
}

export default Main;

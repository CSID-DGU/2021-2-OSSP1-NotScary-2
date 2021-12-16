import React from "react";
import { Component } from "react/cjs/react.production.min";
import { Redirect } from "react-router-dom";
import Webcam from "react-webcam";

class StartCheck extends Component {
  constructor(props) {
    super(props);
    this.state = {
      posture: "", //현재 자세 state
      start: false,
    };
  }

  classPosture = () => {
    this.setState({
      start: true,
    });

    fetch("http://localhost:4000/startcheck", {
      // /posture를 post를 통해 서버와 연동
      method: "post",
      headers: {
        "content-type": "application/json",
      },
    })
      .then((res) => res.json())
      .then((json) => {
        this.setState({
          posture: json.text, //서버로부터 받아온 자세로 현재 state 자세 업데이트
        });
      });
  };

  render() {
    return (
      <div className="start">
        {!this.state.start && (
          <button className="startButton" onClick={this.classPosture}>
            Start
          </button>
        )}

        {this.state.start && (
          <div className="main">
            <div className="loader10" />
            <br />
            <br />
            양쪽 어깨와 눈이 보이도록 설정 후 5초 동안 자세를 유지해주세요.
            <br />
            <br />
            {this.state.posture}
          </div>
        )}
        {this.state.posture == 1 ? (
          <Redirect to={{ pathname: "/pose" }} />
        ) : (
          <></>
        )}
      </div>
    );
  }
}

export default StartCheck;

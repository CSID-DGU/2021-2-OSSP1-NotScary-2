import React from "react";
import { Component } from "react/cjs/react.production.min";
import * as tf from "@tensorflow/tfjs";

class Pose extends Component {
  constructor(props) {
    super(props);
    this.state = {
      posture: "", //현재 자세 state
    };
  }
  //서버로 5가지 좌표값을 통해 분석된 자세 받아오는 함수

  classPosture = setInterval(() => {
    console.log("버튼클릭");
    const post = {
      // //ShoulderDistance,EyeDistance,ShoulderEyeDistance,ShoulderSlope,EyeSlope
      // //SD:-1, ED:-1, SED:-1, SS:-1, ES:-1 로 초기 값 설정해서 진행
      // SD: 418,
      // ED: 117,
      // SED: 188.2727,
      // SS: 0,
      // ES: 0, //임의로 설정한 자세 값(1번 자세)
    };

    fetch("http://localhost:4000/posture", {
      // /posture를 post를 통해 서버와 연동
      method: "post",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify(post),
    })
      .then((res) => res.json())
      .then((json) => {
        this.setState({
          posture: json.text, //서버로부터 받아온 자세로 현재 state 자세 업데이트
        });
      });
  }, 5000);

  render() {
    return (
      <div className="pose">
        {this.state.posture == 0 && "좋은 자세입니다."}
        <div style={{ color: "red" }}>
          {this.state.posture == 1 && "거북목 자세입니다."}
          {this.state.posture == 2 && "턱을 괸 자세입니다."}
        </div>
      </div>
    );
  }
}

export default Pose;

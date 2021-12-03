import React, {useState, useRef} from "react";
import { Component } from "react/cjs/react.production.min";
import { Redirect } from "react-router-dom";
import Webcam from "react-webcam";

const getWebcam = (callback) => {
    try {
      const constraints = {
        'video': true,
        'audio': false
      }
      navigator.mediaDevices.getUserMedia(constraints)
        .then(callback);
    } catch (err) {
      console.log(err);
      return undefined;
    }
  }
  
  const Styles = {
    Video: { width: "100vw", height: "80vh" },
    None: { display: 'none' },
  }

  function StartCheck() {
    const [playing, setPlaying] = React.useState(undefined);
  
    const videoRef = React.useRef(null);
  
    React.useEffect(() => {
      getWebcam((stream => {
        setPlaying(true);
        videoRef.current.srcObject = stream;
      }));
    }, []);
  
    const startOrStop = () => {
      if (playing) {
        const s = videoRef.current.srcObject;
        s.getTracks().forEach((track) => {
          track.stop();
        });
      } else {
        getWebcam((stream => {
          setPlaying(true);
          videoRef.current.srcObject = stream;
        }));
      }
      setPlaying(!playing);
    }
  
    return (<>
      <div style={{ width: '100vw', height: '100vh', padding: '3em' }}>
        <video ref={videoRef} autoPlay style={Styles.Video} />
        <button color="warning" onClick={() => startOrStop()}>{playing ? 'Stop' : 'Start'} </button>
      </div >
    </>);
  }

export default StartCheck;

/*import React from "react";
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
    console.log(this.state);
    

    const videoConstraints = {
      width: 1280,
      height: 720,
      facingMode: "user",
    };

    return (
      <div className="start">
        
        {!this.state.start && (
          <button className="startButton" onClick={this.classPosture}>
            Start
          </button>
        )}

        {this.state.start && (
          <div className="main">
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
*/

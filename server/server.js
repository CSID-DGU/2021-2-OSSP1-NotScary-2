// 시작 자세 체크 서버(StartCheck2.js와 연결)

const express = require("express");
const app = express();
const port = 4000;
const cors = require("cors");
const bodyParser = require("body-parser");

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(cors());

app.post("/startcheck", (req, res) => {
  const { spawn } = require("child_process");

  const result_01 = spawn("python", ["./start_check2.py"]);
  const sendText = { text: "" };
  console.log("시작11");

  result_01.stdout.on("data", (result) => {
    console.log(result.toString());
    sendText.text = result.toString();
    res.send(sendText);
  });

  setTimeout(function () {}, 2000);

  console.log("삭제 시작");
  var fs = require("fs"); // 다운로드된 파일 경로 (개인 pc 경로에 맞게 수정해줘야됨)
  fs.unlink(`C:\\Users\\82109\\Downloads\\start_capture.jpg`, (err) => {
    if (err) console.log(err);
    else {
      console.log("삭제끝");
    }
  });
});

app.listen(port, () => {
  console.log(`Connect at http://localhost:${port}`);
});

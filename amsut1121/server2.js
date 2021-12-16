const express = require("express");
const app = express();
const port = 4001;
const cors = require("cors");
const bodyParser = require("body-parser");

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(bodyParser.json({ limit: "50mb" }));
app.use(
  bodyParser.urlencoded({
    limit: "50mb",
    extended: true,
    parameterLimit: "100000",
  })
);
app.use(cors());

app.post("/pose", (req, res) => {
  // res.header("Access-Control-Allow-Origin", "*");
  // res.set({ "access-control-allow-origin": "*" });

  const { spawn } = require("child_process");
  // const result_01 = spawn("python", ["./webcam_model5.py"]);

  const result_01 = spawn("python", ["./test.py"]);
  const sendText = { text: "" };
  console.log("imagetest 시작");

  result_01.stdout.on("data", (result) => {
    console.log(result.toString());
    sendText.text = result.toString();
    res.send(sendText);
    console.log("imagetest 끝");
  });

  console.log("삭제시작");

  var fs = require("fs"); // 다운로드된 파일 경로 (개인 pc 경로에 맞게 수정해줘야됨)
  fs.unlink(`C:\\Users\\82109\\Downloads\\pose_capture.jpg`, (err) => {
    if (err) console.log(err);
    else {
      console.log("삭제끝");
    }
  });
});

app.listen(port, () => {
  console.log(`Connect at http://localhost:${port}`);
});

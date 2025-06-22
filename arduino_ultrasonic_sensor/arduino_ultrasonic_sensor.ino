/*
  超音波センサー距離測定プログラム
  HC-SR04を使用してPCに距離データを送信します
*/

// ピン定義
const int trigPin = 2;  // トリガーピン
const int echoPin = 3; // エコーピン

void setup() {
  // シリアル通信開始（9600 bps）
  Serial.begin(9600);
  
  // ピンモード設定
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  
  Serial.println("Arduino超音波センサー準備完了");
}

void loop() {
  // 距離測定
  long duration, distance;
  
  // トリガーピンをクリア
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  
  // 10マイクロ秒のパルスを送信
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // エコーピンで戻ってくる時間を測定
  duration = pulseIn(echoPin, HIGH);
  
  // 時間を距離（cm）に変換
  // 音速 = 340m/s = 34000cm/s
  // 往復時間なので2で割る
  distance = duration * 0.034 / 2;
  
  // 有効な範囲の距離のみ送信（2cm-400cm）
  if (distance >= 2 && distance <= 400) {
    Serial.print("Distance: ");
    Serial.println(distance);
  }
  
  // 100ms待機
  delay(100);
} 
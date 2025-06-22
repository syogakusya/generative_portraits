import serial
import time
import sys

def test_arduino_connection():
    """Arduino接続テスト用のスクリプト"""
    
    # COMポート設定（環境に応じて変更してください）
    com_ports = ['COM3', 'COM4', 'COM5', 'COM6']  # 一般的なCOMポート
    
    for com_port in com_ports:
        print(f"COMポート {com_port} への接続を試行中...")
        try:
            # シリアル接続を開く
            arduino = serial.Serial(com_port, 9600, timeout=1)
            time.sleep(2)  # Arduino初期化待機
            
            print(f"✓ {com_port} に接続成功！")
            print("距離データを受信中... (Ctrl+Cで終了)")
            
            # 距離データを受信
            for i in range(20):  # 20回データを受信
                if arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8').strip()
                    if line:
                        print(f"受信データ: {line}")
                time.sleep(0.1)
            
            arduino.close()
            print(f"✓ {com_port} の接続を正常に終了")
            return True
            
        except serial.SerialException:
            print(f"✗ {com_port} への接続に失敗")
            continue
        except Exception as e:
            print(f"✗ エラー: {e}")
            continue
    
    print("\n利用可能なCOMポートが見つかりませんでした。")
    print("以下を確認してください：")
    print("1. ArduinoがPCに接続されているか")
    print("2. Arduinoドライバがインストールされているか")
    print("3. Arduino IDEで正しいCOMポートが認識されているか")
    print("4. 超音波センサーのプログラムが正しくアップロードされているか")
    return False

if __name__ == "__main__":
    try:
        test_arduino_connection()
    except KeyboardInterrupt:
        print("\n\nテストを中断しました。")
        sys.exit(0) 
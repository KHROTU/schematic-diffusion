import http.server
import socketserver
import json
import os
import sys

HOST = "localhost"
PORT = 8766
OUTPUT_FILENAME = "converter_session.json"

class SessionReceiverHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/data':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                self.send_response(200); self.send_header('Access-Control-Allow-Origin', '*'); self.end_headers(); self.wfile.write(b'{"status": "ok"}')
                self.server.should_shutdown = True
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr); self.send_response(500)
        else: self.send_response(404)
    def do_OPTIONS(self):
        self.send_response(204); self.send_header('Access-Control-Allow-Origin', '*'); self.send_header('Access-Control-Allow-Methods', 'POST'); self.send_header('Access-Control-Allow-Headers', 'Content-Type'); self.end_headers()
    def log_message(self, format, *args): pass

class StoppableTCPServer(socketserver.TCPServer):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); self.should_shutdown = False
    def serve_forever(self):
        while not self.should_shutdown: self.handle_request()

if __name__ == "__main__":
    print(f"Starting on http://{HOST}:{PORT}. Waiting for session...", file=sys.stderr)
    httpd = StoppableTCPServer((HOST, PORT), SessionReceiverHandler)
    httpd.serve_forever()
    print("Session received. Shutting down.", file=sys.stderr)
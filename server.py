import socket
import json
import google.generativeai as g_ai
import time

api_key = "AIzaSyB-qE6V29e4fqngFM49AjJzcz3-mypjzzA"
g_ai.configure(api_key=api_key)
gen_model = g_ai.GenerativeModel('gemini-1.5-flash')

def process_client_request(client_conn):
    try:
        incoming_data = client_conn.recv(1024).decode('utf-8')
        if incoming_data:
            request_data = json.loads(incoming_data)
            request_time = time.time()
            try:
                content_response = gen_model.generate_content(request_data['Prompt'])
                content_text = content_response.text if content_response else "No response"
            except Exception as err:
                content_text = f"Error: {str(err)}"
            response_time = time.time()

            response_data = {
                "Prompt": request_data['Prompt'],
                "Message": content_text.strip(),
                "TimeSent": int(request_time),
                "TimeRecvd": int(response_time),
                "Source": "Gemini"
            }
            
            client_conn.send(json.dumps(response_data).encode('utf-8'))
    except Exception as err:
        print(f"Error processing client request: {str(err)}")
    finally:
        client_conn.close()

def initialize_server(host='localhost', port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    while True:
        try:
            client_conn, address = server_socket.accept()
            process_client_request(client_conn)
        except Exception as err:
            print(f"Error accepting client connection: {str(err)}")

if __name__ == "__main__":
    initialize_server()

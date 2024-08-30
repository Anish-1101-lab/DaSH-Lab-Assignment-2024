import socket
import json
import sys

def dispatch_prompt_to_server(prompt_message, identity, server_address='localhost', server_port_number=9999):
    try:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.connect((server_address, server_port_number))

        request_message = json.dumps({"Prompt": prompt_message})
        conn.send(request_message.encode('utf-8'))

        server_response = conn.recv(4096).decode('utf-8')
        response_content = json.loads(server_response)
        
        response_content['ClientID'] = identity

        with open(f'client_{identity}_result.json', 'a') as result_file:
            json.dump(response_content, result_file, indent=4)
            result_file.write('\n')  

        conn.close()
    except Exception as error:
        print(f"Error: {str(error)}")

def handle_file_input(file_path, identity):
    try:
        with open(file_path, "r") as input_file:
            for line in input_file:
                prompt_message = line.strip()
                if not prompt_message:
                    continue
                dispatch_prompt_to_server(prompt_message, identity)
    except Exception as error:
        print(f"Error processing input file: {str(error)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python client.py <input_file> <client_id>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    client_identity = sys.argv[2]
    
    handle_file_input(file_path, client_identity)

from app import create_app

if __name__ == '__main__':
    app = create_app()
    # Para producción, establecer threaded=True para mejor rendimiento con múltiples clientes
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True) 
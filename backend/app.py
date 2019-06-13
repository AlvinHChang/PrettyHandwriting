from flask import Flask
from flask_restful import Api, Resource, reqparse


app = Flask(__name__)
api = Api(app)


class PrettyHandwriting(Resource):

    def get(self, name):
        return name, 200

    def post(self, name):
        parser = reqparse.RequestParser()
        parser.add_argument("age")
        args = parser.parse_args()
        return name, args["age"]


api.add_resource(PrettyHandwriting, '/handwriting/<string:name>')
app.run(debug=True)
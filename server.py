import model 
from flask import Flask, request, render_template, make_response
from flask_restful import Resource, Api


app = Flask(__name__)
api = Api(app)


class index(Resource):
    def get(self):
        return make_response(render_template('index.html'))


class probability(Resource):
    def get(self):
        try:
            age = int(request.args.get('age'))
            workclass = str(request.args.get('work'))
            education = str(request.args.get('education'))
            marital_status= str(request.args.get('status'))
            race = str(request.args.get('race'))
            sex = str(request.args.get('sex'))
            hours = int(request.args.get('hours'))

            prediction = model.predict_probability(age, workclass, education, marital_status, race, sex, hours)

            return make_response(render_template('output.html', prediction=prediction))

        except:
            return 'An Error occurred'


api.add_resource(index, '/')
api.add_resource(probability, '/probability')


if(__name__=='__main__'):
    app.run(debug=True)  
import unittest
from fastapi.testclient import TestClient
from app import app
import boto3

class TestAPI(unittest.TestCase):
    client = TestClient(app)
    data={'sepal_length': 4.0, 
          'sepal_width': 3.0,
        'petal_length': 3.0, 
        'petal_width': 2.0}
    
    def test_reponse(self):
        """
        Vérifie que la reponse est correcte
        """
        reponse=self.client.post("/predict",
                     json=self.data)
        self.assertEqual(reponse.status_code,200)
        self.assertEqual(dict,type(reponse.json()))
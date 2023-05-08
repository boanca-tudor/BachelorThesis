import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class LoginService {
  private baseUrl = 'http://localhost:5000';

  constructor(private http: HttpClient) {}

  sendCredentials(email: string, password: string) {
    return this.http.post(`${this.baseUrl}/login`, {
      email: email,
      password: password,
    });
  }
}

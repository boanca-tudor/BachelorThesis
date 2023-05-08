import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root',
})
export class ResultService {
  private baseUrl = 'http://localhost:5000';

  constructor(private http: HttpClient) {}

  getBaseImage() {
    return this.http.get(`${this.baseUrl}/getUploadedImage`, {
      responseType: 'blob',
    });
  }
}

import { EventEmitter, Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { catchError, Observable, retry, throwError, timeout } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  private apiUrl = 'http://localhost:8222/';
  authStatusChanged = new EventEmitter<void>();

  constructor(private http: HttpClient) {}

  register(userData: any): Observable<any> {

    return this.http.post(`${this.apiUrl}user/signup`, userData);
  }

  login(credentials: any): Observable<any> {
    return this.http.post(`${this.apiUrl}user/login`, credentials);
  }

  getUsers(): Observable<any> {
    return this.http.get(`${this.apiUrl}user/paginate`);
  }

  generate_response(prompt: string): Observable<any> {
    return this.http.post(`${this.apiUrl}ft/generate`, { prompt });
  }
  // In your auth.service.ts
  // auth.service.ts
  threat_detection(prompt: string): Observable<any> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });

    return this.http.post<any>(`${this.apiUrl}ai/detection`, { prompt }, { headers })
      .pipe(
        timeout(300000),  // 5 minutes timeout
        retry(1),
        catchError(error => {
          console.error('API Error:', error);
          throw error;
        })
      );
  }
// Dans le service auth.service.ts
  query_rag(prompt: string): Observable<any> {
    return this.http.post(`${this.apiUrl}rag/generate`, { query_text: prompt });
  }



  saveToken(token: string): void {
    localStorage.setItem('authToken', token);
  }

  getToken(): string | null {
    return localStorage.getItem('authToken');
  }
  removeToken() {
    localStorage.removeItem('token');
    this.authStatusChanged.emit();
  }
}
